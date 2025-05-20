from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import math

from supervision.config import (
    ORIENTED_BOX_COORDINATES,
)
from supervision.utils.file import (
    read_json_file,
)


def xyxyxyxy_to_xyxy(xyxyxyxy: list[list[float]]) -> list[float]:
    """
    Convert a darwin json ellipse annotation to supervision
    axis-aligned bounding box in xyxy format.
    """
    min_x = min(xyxyxyxy[0][0], xyxyxyxy[1][0], xyxyxyxy[2][0], xyxyxyxy[3][0])
    min_y = min(xyxyxyxy[0][1], xyxyxyxy[1][1], xyxyxyxy[2][1], xyxyxyxy[3][1])
    max_x = max(xyxyxyxy[0][0], xyxyxyxy[1][0], xyxyxyxy[2][0], xyxyxyxy[3][0])
    max_y = max(xyxyxyxy[0][1], xyxyxyxy[1][1], xyxyxyxy[2][1], xyxyxyxy[3][1])
    return [min_x, min_y, max_x, max_y]


def darwin_ellipse_to_xyxy(darwin_ellipse: dict) -> list[float]:
    """
    Convert a darwin json ellipse annotation to supervision
    axis-aligned bounding box in xyxy format.
    """
    xyxyxyxy = darwin_ellipse_to_xyxyxyxy(darwin_ellipse)
    xyxy = xyxyxyxy_to_xyxy(xyxyxyxy)
    return xyxy


def darwin_ellipse_to_xyxyxyxy(darwin_ellipse: dict) -> list[list[float]]:
    """
    Convert a darwin json ellipse annotation to supervision
    oriented bounding box in xyxyxyxy format.

    Returns list of 4 xy points
    """
    cx = round(darwin_ellipse["center"]["x"])
    cy = round(darwin_ellipse["center"]["y"])
    rx = round(darwin_ellipse["radius"]["x"])
    ry = round(darwin_ellipse["radius"]["y"])
    angle_rad = darwin_ellipse["angle"]

    # Precompute cos and sin of the angle
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # Define the 4 corners of the bounding rectangle before rotation
    corners = [(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)]

    # Rotate and translate corners
    rotated_corners = []
    for dx, dy in corners:
        x = cx + dx * cos_theta - dy * sin_theta
        y = cy + dx * sin_theta + dy * cos_theta
        rotated_corners.append([x, y])

    return rotated_corners


def darwin_ellipse_to_mask(
    darwin_ellipse: dict, height: int, width: int
) -> npt.NDArray[np.uint8]:
    """
    Creates a binary mask with a filled ellipse.

    Parameters:
    - image_shape: (height, width) of the output mask
    - center: dict with 'x' and 'y'
    - radius: dict with 'x' and 'y'
    - angle_deg: rotation angle in rad

    Returns:
    - mask: NumPy array (dtype=np.uint8) with 1s inside the ellipse and 0s elsewhere
    """
    cx = round(darwin_ellipse["center"]["x"])
    cy = round(darwin_ellipse["center"]["y"])
    rx = round(darwin_ellipse["radius"]["x"])
    ry = round(darwin_ellipse["radius"]["y"])
    angle_rad = darwin_ellipse["angle"]
    angle_deg = np.degrees(angle_rad)

    mask = np.zeros((height, width), dtype=np.uint8)
    center_tuple = (cx, cy)
    axes_tuple = (rx, ry)

    cv2.ellipse(
        mask,
        center=center_tuple,
        axes=axes_tuple,
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    return mask


def empty_mask(height: int, width: int) -> npt.NDArray[np.uint8]:
    """
    Create an empty mask of the given height and width.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        npt.NDArray[np.uint8]: Empty mask array of shape (H, W).
    """
    return np.zeros((height, width), dtype=np.uint8)


def darwin_polygon_to_mask(
    darwin_polygon: dict, height: int, width: int
) -> npt.NDArray[np.uint8]:
    """
    Convert Darwin polygon annotation to mask.

    Args:
        annotation (dict): Annotation dictionary containing polygon data.
        resolution_wh (tuple[int, int]): Resolution of the image (width, height).

    Returns:
        npt.NDArray[np.uint8]: Mask array shape (H, W). Values are 0 or 1.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = []
    for path_points in darwin_polygon["paths"]:  # Darwin 2.0
        polygon_x = [p["x"] for p in path_points]
        polygon_y = [p["y"] for p in path_points]
        polygon = np.array([polygon_x, polygon_y], dtype=np.int32).T
        polygons.append(polygon)

    cv2.fillPoly(mask, polygons, color=1)
    return mask


def darwin_bounding_box_to_xyxy(darwin_bounding_box: dict) -> list[float]:
    """
    Convert darwin dict bounding box format to supervision's xyxy format
    """
    xyxy = [
        darwin_bounding_box["x"],
        darwin_bounding_box["y"],
        darwin_bounding_box["x"] + darwin_bounding_box["w"],
        darwin_bounding_box["y"] + darwin_bounding_box["h"],
    ]
    return xyxy



def darwin_annotations_to_detections_dict(
    json_name: str,
    with_masks: bool,
    classes: list[str],
    with_ellipse_as: Optional[str] = None,
    with_track_ids: bool = False,
    skip_unknown_classes: bool = True,
    metadata: dict = {},
) -> dict:
    """
    Load Darwin annotations from a JSON file and convert them to
    a dictionary with xyxy, class_id and mask keys.

    For users, load Detections with Detections.from_darwin method instead.
    a dictionary with xyxy, class_id, mask, and metadata keys.
    Properties from each annotation are included in the metadata dictionary.

    Args:
        json_name (str): Path to the JSON file containing annotations.
        with_masks (bool): Whether to include masks in the Detections.
        classes (list[str]): List of class names.
        with_ellipse_as (str): How to convert ellipse. Options are
            "oriented_bounding_box", "mask", or None to ignore.
            Default is None.
            "oriented_bounding_box" requires all annotations to be ellipses.
        with_track_ids (bool): Whether to include tracking IDs.
        skip_unknown_classes (bool): Whether to skip unknown classes. Default is True.

    Returns:
        dict: Dictionary containing detection data, including metadata with properties.
    """
    assert with_ellipse_as in [None, "oriented_bounding_box", "mask"], (
        f"ellipse_as must be one of \
        [None, 'oriented_bounding_box', 'mask'], \\ got {with_ellipse_as}"
    )
    if not json_name:
        return dict(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    json_data = read_json_file(json_name)
    height = json_data["item"]["slots"][0]["height"]
    width = json_data["item"]["slots"][0]["width"]

    # required
    xyxy, class_ids = [], []
    # optional depending on arguments
    if with_masks:
        masks = []
    if with_track_ids:
        tracker_id = []
    # who knows, contains properties and optionally oriented bounding boxes
    detection_data = defaultdict(list)

    # update classes with darwin classes, based on skip_unknown_classes
    if not skip_unknown_classes:
        for annotation in json_data["annotations"]:
            if "name" in annotation and annotation["name"] not in classes:
                classes.append(annotation["name"])

    annotations = [
        annotation
        for annotation in json_data["annotations"]
        if annotation.get("name") in classes
    ]

    if with_ellipse_as is not None:
        ellipse_annotations = [ann for ann in annotations if "ellipse" in ann]
        for ellipse_annotation in ellipse_annotations:
            xyxy.append(darwin_ellipse_to_xyxy(ellipse_annotation["ellipse"]))
            class_ids.append(classes.index(ellipse_annotation["name"]))
            if with_track_ids:
                tracker_id.append(
                    ellipse_annotation.get("instance_id", {}).get("value", None)
                )
            if with_ellipse_as == "mask":
                masks.append(
                    darwin_ellipse_to_mask(
                        ellipse_annotation["ellipse"],
                        height=height,
                        width=width,
                    )
                )
            elif with_ellipse_as == "oriented_bounding_box":
                xyxyxyxy = darwin_ellipse_to_xyxyxyxy(ellipse_annotation["ellipse"])
                detection_data[ORIENTED_BOX_COORDINATES].append(xyxyxyxy)
            else:
                raise ValueError(f"Unknown with_ellipse_as value: {with_ellipse_as}")

    normal_annotations = [ann for ann in annotations if "bounding_box" in ann]
    for annotation in normal_annotations:
        xyxy.append(darwin_bounding_box_to_xyxy(annotation["bounding_box"]))
        class_ids.append(classes.index(annotation["name"]))
        detection_data["properties"].append(annotation.get("properties", []))
        if ORIENTED_BOX_COORDINATES in detection_data:
            raise ValueError(
                'Oriented bounding boxes and normal bounding boxes cannot " \
                be used together. If you are trying to use oriented bounding \
                boxes, prevent loading other annotation types by setting classes \
                to your ellipse classes only.'
            )
        if with_masks:
            if "polygon" in annotation:
                masks.append(
                    darwin_polygon_to_mask(
                        annotation["polygon"],
                        height=height,
                        width=width,
                    )
                )
            else:
                masks.append(empty_mask(height, width))
        if with_track_ids:
            tracker_id.append(annotation.get("instance_id", {}).get("value", None))

    # to deal with empty darwin files
    if len(xyxy) == 0:
        xyxy = np.empty((0, 4), dtype=np.float32)
        class_ids = np.array([], dtype=int)
        if with_masks:
            masks = np.empty((0, height, width), dtype=np.uint8)
        if with_track_ids:
            tracker_id = np.empty((0, 1), dtype=np.float32)
    else:
        xyxy = np.asarray(xyxy, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=int)
        if with_masks:
            masks = np.array(masks)
        if with_track_ids:
            tracker_id = np.asarray(tracker_id)
        if ORIENTED_BOX_COORDINATES in detection_data:
            detection_data[ORIENTED_BOX_COORDINATES] = np.array(
                detection_data[ORIENTED_BOX_COORDINATES], dtype=np.float32
            )

    result_dict = {}
    result_dict["xyxy"] = xyxy
    len_xyxy = len(xyxy)
    result_dict["class_id"] = class_ids
    assert len_xyxy == len(class_ids), "xyxy and class_id must have the same length."
    if with_masks:
        result_dict["mask"] = masks
        assert len_xyxy == len(masks), "xyxy and mask must have the same length."
    if with_track_ids:
        result_dict["tracker_id"] = tracker_id
        assert len_xyxy == len(tracker_id), (
            "xyxy and tracker_id must have the same length."
        )
    if len(detection_data) > 0:
        result_dict["data"] = dict(detection_data)
        for key, values in result_dict["data"].items():
            assert len_xyxy == len(values), f"xyxy and {key} must have the same length."

    if len(metadata) > 0:
        result_dict["metadata"] = metadata
    return result_dict


if __name__ == "__main__":
    json_name = "/home/agro/w-drive-vision/GARdata/new_format/ \
        3710496261_broccoli_detection/datasets/anns/ \
        temporaryannottated/annotations/20170815_191925293_RGB.json"
    classes = ["healthy", "damaged", "mature", "cateye", "headrot", "broccoli"]
    temp = darwin_annotations_to_detections_dict(
        json_name,
        with_masks=True,
        classes=classes,
        with_radius=True,
        with_track_ids=True,
    )