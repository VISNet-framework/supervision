from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial

import cv2
import numpy as np
import numpy.typing as npt

from supervision.config import (
    ORIENTED_BOX_COORDINATES,
)
from supervision.utils.file import (
    read_json_file,
)


class AnnotationType(Enum):
    """
    Enum for annotation types.
    """

    ELLIPSE = auto()
    # bounding box may also include polygons
    BOUNDING_BOX = auto()
    UNKNOWN = auto()


def get_annotation_type(annotation: dict) -> AnnotationType:
    """
    Get the type of annotation from the given dictionary.

    Args:
        annotation (dict): Annotation dictionary.

    Returns:
        AnnotationType: The type of annotation.
    """
    if "ellipse" in annotation:
        return AnnotationType.ELLIPSE
    elif "bounding_box" in annotation:
        return AnnotationType.BOUNDING_BOX
    else:
        return AnnotationType.UNKNOWN


@dataclass
class SingleDetection:
    """
    A single detection with its properties.
    Helper class to process darwin annotations and convert them to
    a single supervision Detections object.
    """

    xyxy: list[float]
    class_id: int
    mask: npt.NDArray[np.uint8] | None = None
    tracker_id: int | None = None
    data: dict | None = field(default_factory=dict)


def merge_detections_to_dict(dets: list[SingleDetection]) -> dict:
    """
    Merge a list of SingleDetection objects into a dictionary.
    This is useful for converting darwin annotations to a single
    supervision Detections object.
    """
    xyxy = np.array([d.xyxy for d in dets], dtype=np.float32)
    class_id = np.array([d.class_id for d in dets], dtype=int)

    # All arrays must be equal length, so if a key in data does
    # not exist in a detection, we set it to None
    # TODO check if this leads to issues when trying to use
    # oriented bounding boxes in a dataset with other annotations
    # as well. Easy fix would be to only specify obb classes, but
    # this is now left to the wisdom of the user
    data_keys = {k for d in dets for k in d.data.keys()}
    data = {k: [] for k in data_keys}
    for d in dets:
        for key in data_keys:
            data[key].append(d.data.get(key, None))
    try:
        data = {k: np.array(v) for k, v in data.items()}
    except ValueError:
        data = {k: list(v) for k, v in data.items()}

    result = {
        "xyxy": xyxy,
        "class_id": np.array(class_id),
        "data": data,
    }

    tracker_ids = np.array(
        [d.tracker_id for d in dets if d.tracker_id is not None], dtype=int
    )
    if len(tracker_ids) > 0:
        result["tracker_id"] = tracker_ids

    masks = np.array([d.mask for d in dets if d.mask is not None], dtype=np.uint8)
    if len(masks) > 0:
        result["mask"] = masks
    return result


def process_ellipse_as_mask(
    annotation: dict, classes: list[str], width: int, height: int, with_track_ids: bool
) -> SingleDetection:
    """
    Process a single ellipse annotation and convert it to a SingleDetection object.
    """
    xyxy = darwin_ellipse_to_xyxy(annotation["ellipse"])
    class_id = classes.index(annotation["name"])
    mask = darwin_ellipse_to_mask(
        annotation["ellipse"],
        height=height,
        width=width,
    )
    data = {}
    if "properties" in annotation:
        # NOTE this will probably work, but might not be ideal
        # darwin stores so-called properties as list of dicts
        # per dict a key and a value
        # might be simpler to directly add the key and value to data
        # otherwise, properties will be a random bag of attributes within data
        # which is already the same task as data itself
        data["properties"] = annotation["properties"]
    if with_track_ids:
        tracker_id = annotation.get("instance_id", {}).get("value", None)
    else:
        tracker_id = None
    return SingleDetection(
        xyxy=xyxy, class_id=class_id, mask=mask, tracker_id=tracker_id, data=data
    )


def process_ellipse_as_obb(
    annotation: dict, classes: list[str], with_track_ids: bool
) -> SingleDetection:
    """
    Process a single ellipse annotation and convert it to a SingleDetection object.
    """
    xyxyxyxy = darwin_ellipse_to_xyxyxyxy(annotation["ellipse"])
    xyxy = xyxyxyxy_to_xyxy(xyxyxyxy)
    class_id = classes.index(annotation["name"])
    data = {ORIENTED_BOX_COORDINATES: xyxyxyxy}
    if "properties" in annotation:
        # NOTE this will probably work, but might not be ideal
        # darwin stores so-called properties as list of dicts
        # per dict a key and a value
        # might be simpler to directly add the key and value to data
        # otherwise, properties will be a random bag of attributes within data
        # which is already the same task as data itself
        data["properties"] = annotation["properties"]
    if with_track_ids:
        tracker_id = annotation.get("instance_id", {}).get("value", None)
    else:
        tracker_id = None
    return SingleDetection(
        xyxy=xyxy, class_id=class_id, data=data, tracker_id=tracker_id
    )


def process_bounding_box(
    annotation: dict,
    classes: list[str],
    width: int,
    height: int,
    with_masks: bool,
    with_track_ids: bool,
) -> SingleDetection:
    """
    Process a single bounding box annotation and convert it to a SingleDetection object.
    """
    xyxy = darwin_bounding_box_to_xyxy(annotation["bounding_box"])
    class_id = classes.index(annotation["name"])
    data = {}
    if "properties" in annotation:
        # NOTE this will probably work, but might not be ideal
        # darwin stores so-called properties as list of dicts
        # per dict a key and a value
        # might be simpler to directly add the key and value to data
        # otherwise, properties will be a random bag of attributes within data
        # which is already the same task as data itself
        data["properties"] = annotation["properties"]
    if with_track_ids:
        tracker_id = annotation.get("instance_id", {}).get("value", None)
    else:
        tracker_id = None

    if with_masks:
        if "polygon" in annotation:
            mask = darwin_polygon_to_mask(
                annotation["polygon"],
                height=height,
                width=width,
            )
        else:
            mask = empty_mask(height, width)
    else:
        mask = None
    return SingleDetection(
        xyxy=xyxy, class_id=class_id, mask=mask, tracker_id=tracker_id, data=data
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
    # left up, right up, right down, left, down (in image space)
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
    with_ellipse_as: str | None = None,
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

    annotation_parser = {
        AnnotationType.BOUNDING_BOX.value: partial(
            process_bounding_box,
            width=width,
            height=height,
            classes=classes,
            with_masks=with_masks,
            with_track_ids=with_track_ids,
        )
    }

    if with_ellipse_as == "mask":
        annotation_parser[AnnotationType.ELLIPSE.value] = partial(
            process_ellipse_as_mask,
            width=width,
            height=height,
            classes=classes,
            with_track_ids=with_track_ids,
        )
    elif with_ellipse_as == "oriented_bounding_box":
        annotation_parser[AnnotationType.ELLIPSE.value] = partial(
            process_ellipse_as_obb, classes=classes, with_track_ids=with_track_ids
        )
    elif with_ellipse_as is None:
        pass
    else:
        raise ValueError(
            f"with_ellipse_as must be one of [None, 'oriented_bounding_box', 'mask'], \
                got {with_ellipse_as}"
        )

    single_detections = []
    for annotation in annotations:
        annotation_type = get_annotation_type(annotation)
        if annotation_type.value in annotation_parser:
            single_detections.append(
                annotation_parser[annotation_type.value](annotation)
            )

    result_dict = merge_detections_to_dict(single_detections)
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
