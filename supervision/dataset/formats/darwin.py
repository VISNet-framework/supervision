import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from natsort import natsorted

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
    approximation_percentage: float = 0.75,
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
            Percentage of points for polygon approximation. Default is 0.75.
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
    approximation_percentage: float = 0.75,
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
    for xyxy, mask, confidence, class_id, tracker_id, data in detections:
        item_id = _generate_item_id()

        x1y1wh = xyxy.copy()
        x1y1wh[2] = xyxy[2] - xyxy[0]
        x1y1wh[3] = xyxy[3] - xyxy[1]

        # TODO ask bart if we need to check if bbox is inside image
        # x1y1wh[0] = max(x1y1wh[0], 0)
        # x1y1wh[1] = max(x1y1wh[1], 0)

        assert not x1y1wh[0] < 0, f"Negative x1 {x1y1wh[0]} in {image_filename}"
        assert not x1y1wh[1] < 0, f"Negative y1 {x1y1wh[1]} in {image_filename}"
        assert not x1y1wh[2] < 0, f"Negative w {x1y1wh[2]} in {image_filename}"
        assert not x1y1wh[3] < 0, f"Negative h {x1y1wh[3]} in {image_filename}"

        bbox = {"h": x1y1wh[3], "w": x1y1wh[2], "x": x1y1wh[0], "y": x1y1wh[1]}
        class_name = classes[int(class_id)]

        annotation = {
            "bounding_box": bbox,
            "id": str(item_id),
            "instance_id": {"value": tracker_id},
            "name": class_name,
            "properties": data.get("properties", [[]])[0],
            "slot_names": ["0"],
            "score": str(confidence),
        }
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
