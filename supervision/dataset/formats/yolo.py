from __future__ import annotations

import os
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.utils import approximate_mask_with_polygons
from supervision.detection.core import Detections
from supervision.detection.utils.converters import polygon_to_mask, polygon_to_xyxy
from supervision.utils.file import (
    list_files_with_extensions,
    read_txt_file,
    read_yaml_file,
    save_text_file,
    save_yaml_file,
)
from supervision.utils.image import load_image_shape_quick

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset


def _parse_box(values: list[str]) -> np.ndarray:
    x_center, y_center, width, height = values
    return np.array(
        [
            float(x_center) - float(width) / 2,
            float(y_center) - float(height) / 2,
            float(x_center) + float(width) / 2,
            float(y_center) + float(height) / 2,
        ],
        dtype=np.float32,
    )


def _box_to_polygon(box: np.ndarray) -> np.ndarray:
    return np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )


def _parse_polygon(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def _polygons_to_masks(
    polygons: list[np.ndarray], resolution_wh: tuple[int, int]
) -> np.ndarray:
    return np.array(
        [
            polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
            for polygon in polygons
        ],
        dtype=bool,
    )


def _with_mask(lines: list[str]) -> bool:
    return any([len(line.split()) > 5 for line in lines])


def _extract_class_names(file_path: str) -> list[str]:
    data = read_yaml_file(file_path=file_path)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names.keys())]
    return names


def _relative_image_path(image_path: str, image_directory_name="images"):
    """
    Returns the relative path starting from a directory name.

    Default image base directory is "images"
    """
    images_dirname = os.path.sep + image_directory_name + os.path.sep
    relative_path_image = image_path.split(images_dirname)[-1]
    return relative_path_image


def _image_path_to_annotation_path(image_path: str) -> str:
    """
    Returns the yolo-style annotation path.

    Note that ultralytics finds annotations by replacing "/images/" with "/annotations/"
    For nested image directories, the annotations should be stored in a nested
    directory as well.
    """
    relative_path_image = _relative_image_path(image_path)
    base_name, _ = os.path.splitext(relative_path_image)
    relative_path_annotation = base_name + ".txt"
    return relative_path_annotation


def yolo_annotations_to_detections(
    lines: list[str],
    resolution_wh: tuple[int, int],
    with_masks: bool,
    is_obb: bool = False,
) -> Detections:
    if len(lines) == 0:
        return Detections.empty()

    class_id, relative_xyxy, relative_polygon, relative_xyxyxyxy = [], [], [], []
    w, h = resolution_wh
    for line in lines:
        values = line.split()
        class_id.append(int(values[0]))
        if len(values) == 5:
            box = _parse_box(values=values[1:])
            relative_xyxy.append(box)
            if with_masks:
                relative_polygon.append(_box_to_polygon(box=box))
        elif len(values) > 5:
            polygon = _parse_polygon(values=values[1:])
            relative_xyxy.append(polygon_to_xyxy(polygon=polygon))
            if is_obb:
                relative_xyxyxyxy.append(np.array(values[1:]))
            if with_masks:
                relative_polygon.append(polygon)

    class_id = np.array(class_id, dtype=int)
    relative_xyxy = np.array(relative_xyxy, dtype=np.float32)
    xyxy = relative_xyxy * np.array([w, h, w, h], dtype=np.float32)
    data = {}

    if is_obb:
        relative_xyxyxyxy = np.array(relative_xyxyxyxy, dtype=np.float32)
        xyxyxyxy = relative_xyxyxyxy.reshape(-1, 4, 2)
        xyxyxyxy *= np.array([w, h], dtype=np.float32)
        data[ORIENTED_BOX_COORDINATES] = xyxyxyxy

    if not with_masks:
        return Detections(class_id=class_id, xyxy=xyxy, data=data)

    polygons = [
        (polygon * np.array(resolution_wh)).astype(int) for polygon in relative_polygon
    ]
    mask = _polygons_to_masks(polygons=polygons, resolution_wh=resolution_wh)
    return Detections(class_id=class_id, xyxy=xyxy, data=data, mask=mask)


def load_yolo_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
    is_obb: bool = False,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images,
        and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_directory_path (str): The path to the directory
            containing the YOLO annotation files.
        data_yaml_path (str): The path to the data
            YAML file containing class information.
        force_masks (bool): If True, forces masks to be loaded
            for all annotations, regardless of whether they are present.
        is_obb (bool): If True, loads the annotations in OBB format.
            OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
            where pairs of [x, y] are box corners.

    Returns:
        Tuple[List[str], List[str], Dict[str, Detections]]:
            A tuple containing a list of class names, a dictionary with
            image names as keys and images as values, and a dictionary
            with image names as keys and corresponding Detections instances as values.
    """
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=images_directory_path,
            extensions=[
                "bmp",
                "dng",
                "jpg",
                "jpeg",
                "mpo",
                "png",
                "tif",
                "tiff",
                "webp",
            ],
        )
    ]

    classes = _extract_class_names(file_path=data_yaml_path)
    annotations = {}

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.txt")
        if not os.path.exists(annotation_path):
            annotations[image_path] = Detections.empty()
            continue

        # PIL is much faster than cv2 for checking image shape and mode: https://github.com/roboflow/supervision/issues/1554
        image = Image.open(image_path)
        lines = read_txt_file(file_path=annotation_path, skip_empty=True)
        w, h = image.size
        resolution_wh = (w, h)
        if image.mode not in ("RGB", "L"):
            raise ValueError(
                f"Images must be 'RGB' or 'grayscale', \
                but {image_path} mode is '{image.mode}'."
            )

        with_masks = _with_mask(lines=lines)
        with_masks = force_masks if force_masks else with_masks
        annotation = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
            is_obb=is_obb,
        )
        annotations[image_path] = annotation
    return classes, image_paths, annotations


def object_to_yolo(
    xyxy: np.ndarray,
    class_id: int,
    image_shape: tuple[int, int, int],
    polygon: np.ndarray | None = None,
) -> str:
    h, w, _ = image_shape
    if polygon is None:
        xyxy_relative = xyxy / np.array([w, h, w, h], dtype=np.float32)
        x_min, y_min, x_max, y_max = xyxy_relative
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return f"{int(class_id)} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}"
    else:
        polygon_relative = polygon / np.array([w, h], dtype=np.float32)
        polygon_relative = polygon_relative.reshape(-1)
        polygon_parsed = " ".join([f"{value:.5f}" for value in polygon_relative])
        return f"{int(class_id)} {polygon_parsed}"


def detections_to_yolo_annotations(
    detections: Detections,
    image_shape: tuple[int, int, int],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> list[str]:
    annotation = []
    for xyxy, mask, _, class_id, _, _ in detections:
        if class_id is None:
            raise ValueError("Class ID is required for YOLO annotations.")

        if mask is not None:
            polygons = approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
            # for polygon in polygons:
            polygon = merge_multi_segment(polygons)
            xyxy = polygon_to_xyxy(polygon=polygon)
            next_object = object_to_yolo(
                xyxy=xyxy,
                class_id=class_id,
                image_shape=image_shape,
                polygon=polygon,
            )
            annotation.append(next_object)
        else:
            next_object = object_to_yolo(
                xyxy=xyxy, class_id=class_id, image_shape=image_shape
            )
            annotation.append(next_object)
    return annotation


def save_yolo_annotations(
    dataset: DetectionDataset,
    annotations_directory_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)

    with ThreadPool() as pool:
        pool.map(
            partial(
                save_yolo_annotation,
                dataset=dataset,
                annotations_directory_path=annotations_directory_path,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            ),
            range(len(dataset)),
        )
    return


def save_yolo_annotation(
    index: int,
    dataset: DetectionDataset,
    annotations_directory_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    image_path = dataset.image_paths[index]
    image_shape = load_image_shape_quick(image_path)
    annotation = dataset.annotations[image_path]

    yolo_annotations_path_rel = _image_path_to_annotation_path(image_path=image_path)
    yolo_annotations_path_abs = (
        Path(annotations_directory_path) / yolo_annotations_path_rel
    )
    yolo_annotations_path_abs.parent.mkdir(exist_ok=True, parents=True)
    lines = detections_to_yolo_annotations(
        detections=annotation,
        image_shape=image_shape,  # type: ignore
        min_image_area_percentage=min_image_area_percentage,
        max_image_area_percentage=max_image_area_percentage,
        approximation_percentage=approximation_percentage,
    )
    save_text_file(lines=lines, file_path=yolo_annotations_path_abs)
    return


def save_data_yaml(data_yaml_path: str, classes: list[str]) -> None:
    data = {"nc": len(classes), "names": classes}
    Path(data_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    save_yaml_file(data=data, file_path=data_yaml_path)


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        idx1 (int): Index of the point in arr1 with the shortest distance.
        idx2 (int): Index of the point in arr2 with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """
    Merge multiple segments into one list by connecting the coordinates
    with the minimum distance between each segment.

    This function connects these coordinates with a thin line to merge all segments.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like
                               [segmentation1, segmentation2, ...].

    Returns:
        s (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s
