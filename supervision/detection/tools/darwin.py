from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from supervision.utils.file import (
    read_json_file,
)


def darwin_annotations_to_masks(
    annotation: dict,
    resolution_wh: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """
    Convert Darwin annotation to mask.

    Args:
        annotation (dict): Annotation dictionary containing polygon data.
        resolution_wh (tuple[int, int]): Resolution of the image (width, height).

    Returns:
        npt.NDArray[np.uint8]: Mask array shape (H, W). Values are 0 or 1.
    """
    assert "polygon" in annotation, (
        f"Annotation {annotation} does not contain polygon data"
    )
    width, height = map(int, resolution_wh)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = []
    for path_points in annotation["polygon"]["paths"]:  # Darwin 2.0
        polygon_x = [p["x"] for p in path_points]
        polygon_y = [p["y"] for p in path_points]
        points = np.array([polygon_x, polygon_y], dtype=np.int32).T
        polygons.append(points)

    cv2.fillPoly(mask, polygons, color=1)
    return mask


def darwin_annotations_to_detections_dict(
    json_name: str,
    with_masks: bool,
    classes: list[str],
    skip_unknown_classes: bool = True,
) -> dict:
    """
    Load Darwin annotations from a JSON file and convert them to
    a dictionary with xyxy, class_id and mask keys.
    load Detections with Detections.from_darwin method.

    Args:
        json_name (str): Path to the JSON file containing annotations.
        with_masks (bool): Whether to include masks in the Detections.
        classes (dict): List of class names.
        skip_unknown_classes (bool): Whether to skip unknown classes. Default is True.
    """
    if not json_name:
        return dict(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    data = read_json_file(json_name)
    height = data["item"]["slots"][0]["height"]
    width = data["item"]["slots"][0]["width"]

    xyxy, class_ids, masks = [], [], []
    for annotation in data["annotations"]:
        if "bounding_box" in annotation:
            if skip_unknown_classes and annotation["name"] not in classes:
                print(f"skipping {annotation}")
                continue

            assert annotation["name"] in classes, f"Unknown class {annotation['name']}"

            class_id = classes.index(annotation["name"])
            class_ids.append(class_id)

            xyxy.append(
                [
                    annotation["bounding_box"]["x"],
                    annotation["bounding_box"]["y"],
                    annotation["bounding_box"]["x"] + annotation["bounding_box"]["w"],
                    annotation["bounding_box"]["y"] + annotation["bounding_box"]["h"],
                ]
            )

            if with_masks:
                mask = darwin_annotations_to_masks(
                    annotation=annotation,
                    resolution_wh=(width, height),
                )
                masks.append(mask)
    ## to deal with empty darwin files
    if xyxy==[]:
        xyxy = np.empty((0, 4), dtype=np.float32)
        class_ids = np.array([], dtype=int)
        if with_masks:
            masks = np.empty((0, int(data["item"]["slots"][0]["height"]), int(data["item"]["slots"][0]["width"])), dtype=np.uint8)
    
    xyxy = np.asarray(xyxy)
    class_ids = np.asarray(class_ids, dtype=int)
    assert xyxy.shape[0] == len(class_ids), (
        f"xyxy len {xyxy.shape[0]}, but class_ids len {len(class_ids)}."
    )
    if with_masks:
        masks = np.array(masks)
        assert masks.shape[0] == len(class_ids), (
            f"masks len {len(masks)}, but class_ids len {len(class_ids)}."
        )
        return dict(xyxy=xyxy, class_id=class_ids, mask=masks)
    else:
        return dict(xyxy=xyxy, class_id=class_ids)
