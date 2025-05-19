import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from natsort import natsorted
import cv2

from supervision.dataset.utils import (
    approximate_mask_with_polygons,
)
from supervision.detection.core import Detections
from supervision.utils.file import (
    list_files_with_extensions_recursively,
    save_json_file,
)

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset


def find_valid_images_and_annotations(
    images_directory_path: Path, annotation_path: Path
) -> Tuple[List[Path], List[Path]]:
    """
    Find valid images and darwin annotations in the given directories.
    """
    image_candidate_paths = list_files_with_extensions_recursively(
        directory=images_directory_path,
        extensions=["jpg", "jpeg", "png", "tiff", "tif"],
    )
    image_candidate_stems = [path.stem for path in image_candidate_paths]
    assert len(image_candidate_stems) == len(set(image_candidate_stems)), (
        "Image filenames must be unique"
    )

    annotation_paths = list_files_with_extensions_recursively(
        directory=annotation_path,
        extensions=["json"],
    )
    annotation_paths = natsorted(annotation_paths)

    image_paths = []
    for annotation_path in annotation_paths:
        # find the corresponding image path
        image_stem = annotation_path.stem
        image_path = image_candidate_paths[image_candidate_stems.index(image_stem)]
        image_paths.append(image_path)
    return image_paths, annotation_paths


def load_darwin_annotations(
    images_directory_path: str | Path,
    annotation_directory_path: str | Path,
    classes: list[str],
    force_masks: bool = False,
    force_track_ids: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Detections]]:
    """
    Load Darwin annotations from a directory.

    Args:
        images_directory_path (str | Path): Path to the directory containing images.
        annotation_directory_path (str | Path):
            Path to the directory containing darwin annotations.
        classes (list[str]): List of class names.
        force_masks (bool): Whether to force loading masks. Default is False.
    """
    ## TODO implement loading of metadata using json:
    # current idea at image location replace.png/.jpg with .json

    images_directory_path = Path(images_directory_path)
    annotation_directory_path = Path(annotation_directory_path)
    images_paths, annotation_paths = find_valid_images_and_annotations(
        images_directory_path=images_directory_path,
        annotation_path=annotation_directory_path,
    )
    images = []
    annotations = {}

    for img_name, annot_name in zip(images_paths, annotation_paths):
        annotation = Detections.from_darwin(
            json_name=annot_name,
            with_masks=force_masks,
            classes=classes,
            skip_unknown_classes=True,
            with_track_ids = force_track_ids,
            # metadata = utils.load_json(img_name.parent / (img_name.stem + ".json")
        )
        images.append(str(img_name))
        annotations[str(img_name)] = annotation

    return classes, images, annotations


def save_darwin_annotations(
    dataset: "DetectionDataset",
    annotation_directory_path: str | Path,
    darwin_dataset_name: str,
    classes: list[str],
    darwin_folder: str = "",
    tags: list = [],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.0,
) -> None:
    """
    Save Darwin annotations to a directory.

    Args:
        dataset (DetectionDataset): Detection dataset containing images and annotations.
        annotation_directory_path (str):
            Path to the directory where annotations will be saved.
        darwin_dataset_name (str): Name of the Darwin dataset.
        classes (list[str]): List of class names.
        darwin_folder (str): Path to the Darwin folder. Default is empty string.
        tags (list): List of tags to add to the annotations. Default is empty list.
        min_image_area_percentage (float):
            Minimum area percentage polygon wrt image. Default is 0.0.
        max_image_area_percentage (float):
            Maximum area percentage polygon wrt image. Default is 1.0.
        approximation_percentage (float):
            Percentage of points for polygon approximation. Default is 0.0 (keep all polygon).
    """
    annotation_directory_path = Path(annotation_directory_path)
    annotation_directory_path.mkdir(parents=True, exist_ok=True)
    for image_path, image, annotation in dataset:
        image_filename = Path(image_path).name
        darwin_annotation_name = Path(image_path).stem + ".json"
        darwin_annotation_path = os.path.join(
            annotation_directory_path, darwin_annotation_name
        )
        annotation_dict = detections_to_darwin_annotations(
            detections=annotation,
            image_shape=image.shape,
            image_filename=Path(image_filename),
            darwin_dataset_name=darwin_dataset_name,
            classes=classes,
            darwin_folder=darwin_folder,
            tags=tags,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )
        save_json_file(
            file_path=darwin_annotation_path,
            data=annotation_dict,
        )
    return


def detection_mask_to_darwin(annotation, mask, min_image_area_percentage, max_image_area_percentage, approximation_percentage):
    if mask is not None:
        # polygons = mask_to_polygons(mask)
        polygons = approximate_mask_with_polygons(
            mask=mask,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        paths = []
        for poly in polygons:
            poly = np.reshape(poly, (-1, 2))
            paths.append([{"x": int(x), "y": int(y)} for x, y in poly])

        annotation["polygon"] = {"paths": paths}
    return annotation


def detection_box_to_darwin(xyxy, classes, class_id, item_id, mask=None, tracker_id=None, properties=[], confidence=None):
    x1y1wh = xyxy.copy()
    x1y1wh[2] = xyxy[2] - xyxy[0]
    x1y1wh[3] = xyxy[3] - xyxy[1]

    # TODO ask bart if we need to check if bbox is inside image -> BART please do not include this because original darwin boxes also can be negative!
    # or lager than image size
    # x1y1wh[0] = max(x1y1wh[0], 0)
    # x1y1wh[1] = max(x1y1wh[1], 0)

    # assert not x1y1wh[0] < 0, f"Negative x1 {x1y1wh[0]} in {image_filename}"
    # assert not x1y1wh[1] < 0, f"Negative y1 {x1y1wh[1]} in {image_filename}"
    # assert not x1y1wh[2] < 0, f"Negative w {x1y1wh[2]} in {image_filename}"
    # assert not x1y1wh[3] < 0, f"Negative h {x1y1wh[3]} in {image_filename}"

    bbox = {
        "h": float(x1y1wh[3]),
        "w": float(x1y1wh[2]),
        "x": float(x1y1wh[0]),
        "y": float(x1y1wh[1]),
    }
    class_name = classes[int(class_id)]

    annotation = {
        "bounding_box": bbox,
        "id": str(item_id),
        "instance_id": {"value": int(tracker_id)},
        "name": class_name,
        "properties": properties, #detections.metadata.get(i, {"properties": [[]]}).get("properties", [[]])[0],
        "slot_names": ["0"],
        # "score": str(confidence),
    }
    if confidence is not None:
        annotation["score"] = str(float(confidence))
    return annotation


def xyxyxyxy_to_darwin_ellipse(xyxyxyxy):
    """
    Estimate Darwin ellipse parameters from 4 points (xyxyxyxy).
    Assumes the points are ordered as in darwin_ellipse_to_xyxyxyxy.

    Args:
        xyxyxyxy (list or np.ndarray): List of 4 [x, y] points.

    Returns:
        dict: {
            "center": {"x": float, "y": float},
            "radius": {"x": float, "y": float},
            "angle": float (radians)
        }
    """
    pts = np.asarray(xyxyxyxy, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("xyxyxyxy must be a list of 4 [x, y] points.")

    # Use OpenCV to fit a rotated rectangle, then extract ellipse parameters
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle_deg = rect

    # OpenCV angle is in degrees, and refers to the rectangle's orientation
    # Convert to radians
    angle_rad = np.deg2rad(angle_deg)

    # The rectangle's width and height correspond to the ellipse's axes
    rx = w / 2.0
    ry = h / 2.0

    # Return in Darwin format
    return {
        "center": {"x": float(cx), "y": float(cy)},
        "radius": {"x": float(rx), "y": float(ry)},
        "angle": float(angle_rad)
    }


def detection_ellipse_to_darwin(xyxyxyxy, classes, class_id, item_id, mask=None, tracker_id=None, properties=[], confidence=None):
    class_name = classes[int(class_id)]

    annotation = {
        "ellipse": xyxyxyxy_to_darwin_ellipse(xyxyxyxy),
        "id": str(item_id),
        "instance_id": {"value": int(tracker_id)},
        "name": class_name,
        "properties": properties, #detections.metadata.get(i, {"properties": [[]]}).get("properties", [[]])[0],
        "slot_names": ["0"],
        # "score": str(confidence),
    }
    if confidence is not None:
        annotation["score"] = str(float(confidence))
    return annotation


def detections_to_darwin_annotations(
    detections: Detections,
    image_shape: tuple[int, int],
    image_filename: Path,
    classes: list[str],
    darwin_dataset_name: str,
    darwin_folder: Path = Path(""),
    tags: list = [],
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.0,
):
    """
    Convert detections to Darwin annotations.
    Args:
        detections (Detections): Detections object containing bounding boxes, masks, etc
        image_shape (tuple[int, int]): Shape of the image (height, width).
        img_filename (Path): Filename of the image.
        class_names (list[str]): List of class names.
        darwin_dataset_name (str): Name of the Darwin dataset.
        darwin_folder (Path): Path to the Darwin folder. Default is empty Path.
        tags (list): List of tags to add to the annotations. Default is empty list.
        min_image_area_percentage (float): Min polygon area wrt image.
        max_image_area_percentage (float): Max polygon area wrt image.
        approximation_percentage (float): Percentage of points for poly approximation.
    """

    def _generate_item_id():
        item_id = "".join(random.choices("abcdef0123456789", k=8))
        item_id = item_id + "-" + "".join(random.choices("abcdef0123456789", k=4))
        item_id = item_id + "-" + "".join(random.choices("abcdef0123456789", k=4))
        item_id = item_id + "-" + "".join(random.choices("abcdef0123456789", k=4))
        item_id = item_id + "-" + "".join(random.choices("abcdef0123456789", k=12))
        return item_id

    height, width = image_shape[0], image_shape[1]

    item_id = _generate_item_id()

    writedata = {}
    writedata["version"] = "2.0"
    writedata["schema_ref"] = (
        "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json_2_0.schema.json"
    )
    writedata["item"] = {
        "name": image_filename.name,
        "path": str(darwin_folder),
        "source_info": {
            "item_id": str(item_id),
            "dataset": {
                "name": darwin_dataset_name,
                "slug": darwin_dataset_name.lower(),
                "dataset_management_url": "https://darwin.v7labs.com/",
            },
            "team": {
                "name": "WUR AgroFoodRobotics",
                "slug": "wur-agrofoodrobotics",
            },
            "workview_url": "https://darwin.v7labs.com/",
        },
        "slots": [
            {
                "type": "image",
                "slot_name": "0",
                "width": width,
                "height": height,
                "thumbnail_url": "",
                "source_files": [
                    {
                        "file_name": image_filename.name,
                        "url": "https://darwin.v7labs.com/",
                        "local_path": str(image_filename),
                    }
                ],
            }
        ],
    }
    writedata["annotations"] = []
    all_properties = detections.data.get("properties", None)
    for i, (xyxy, mask, confidence, class_id, tracker_id, data) in enumerate(detections):
        item_id = _generate_item_id()
        # get properties
        properties = None
        if all_properties is not None:
            properties = all_properties[i]
        
        ## only for xyxyxyxy (oriented bounding boxes)
        if data.get("xyxyxyxy", None) is not None:
            if data["xyxyxyxy"][0] is not None:
                annotation = detection_ellipse_to_darwin(data["xyxyxyxy"][0],
                                                         classes, class_id, item_id, mask, tracker_id, properties, confidence)
                writedata["annotations"].append(annotation)
                continue
        ## TODO add keypoints annotations

        ## target  is bounding box format
        annotation = detection_box_to_darwin(xyxy, classes, class_id, item_id, mask, tracker_id, properties, confidence)
        annotation = detection_mask_to_darwin(annotation, mask, min_image_area_percentage, max_image_area_percentage, approximation_percentage)
        writedata["annotations"].append(annotation)


    for t in tags:
        writedata["annotations"].append(
            {
                "id": str(item_id),
                "name": t,
                "properties": [],
                "slot_names": ["0"],
                "tag": {},
            }
        )
    return writedata
