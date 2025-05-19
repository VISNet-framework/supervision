from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import math

from supervision.utils.file import (
    read_json_file,
)


def darwin_ellipse_to_xyxyxyxy(center, radius, angle_rad):
    cx, cy = center['x'], center['y']
    rx, ry = radius['x'], radius['y']
    theta = angle_rad

    # Precompute cos and sin of the angle
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Define the 4 corners of the bounding rectangle before rotation
    corners = [
        (-rx, -ry),
        ( rx, -ry),
        ( rx,  ry),
        (-rx,  ry)
    ]

    # Rotate and translate corners
    rotated_corners = []
    for dx, dy in corners:
        x = cx + dx * cos_theta - dy * sin_theta
        y = cy + dx * sin_theta + dy * cos_theta
        rotated_corners.append([x, y])

    return rotated_corners


def darwin_ellipse_to_mask(height, width, center, radius, angle_rad):
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
    mask = np.zeros((height, width), dtype=np.uint8)

    center_tuple = (int(round(center['x'])), int(round(center['y'])))
    axes_tuple = (int(round(radius['x'])), int(round(radius['y'])))

    cv2.ellipse(mask, center=center_tuple, axes=axes_tuple,
                angle=np.degrees(angle_rad), startAngle=0, endAngle=360,
                color=1, thickness=-1)
    
    return mask 


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


def get_class_id(annotation, classes, skip_unknown_classes):
    if skip_unknown_classes and annotation["name"] not in classes:
        print(f'skipping {annotation}\nClass {annotation["name"]} unknown')
        return False

    assert annotation["name"] in classes, f"Unknown class {annotation['name']}"

    class_id = classes.index(annotation["name"])
    # class_ids.append(class_id)
    return class_id

def add_box(annotation, with_masks, width, height):

    xyxy =         [
            annotation["bounding_box"]["x"],
            annotation["bounding_box"]["y"],
            annotation["bounding_box"]["x"] + annotation["bounding_box"]["w"],
            annotation["bounding_box"]["y"] + annotation["bounding_box"]["h"],
        ]

    mask= None
    if with_masks:
        mask = darwin_annotations_to_masks(
            annotation=annotation,
            resolution_wh=(width, height),
        )
    return xyxy, mask


def add_ellipse(annotation, with_masks, width, height):
    """
    Extracts bounding box and mask for an ellipse annotation.

    Args:
        annotation (dict): Annotation dictionary containing ellipse data.
        with_masks (bool): Whether to generate a mask.
        width (int): Image width.
        height (int): Image height.

    Returns:
        tuple: (xyxy_box, mask)
    """
    cx = annotation["ellipse"]["center"]["x"]
    cy = annotation["ellipse"]["center"]["y"]
    rx = annotation["ellipse"]["radius"]["x"]
    ry = annotation["ellipse"]["radius"]["y"]

    xyxy_box = [
        cx - rx,
        cy - ry,
        cx + rx,
        cy + ry,
    ]

    mask = None
    if with_masks:
        mask = darwin_ellipse_to_mask(
            height=height,
            width=width,
            center=annotation["ellipse"]["center"],
            radius=annotation["ellipse"]["radius"],
            angle_rad=annotation["ellipse"]["angle"],
        )
    return xyxy_box, mask


def darwin_annotations_to_detections_dict(
    json_name: str,
    with_masks: bool,
    classes: list[str],
    with_track_ids: bool = False,
    skip_unknown_classes: bool = True,
    metadata: dict={}
) -> dict:
    """
    Load Darwin annotations from a JSON file and convert them to
    a dictionary with xyxy, class_id, mask, and metadata keys.
    Properties from each annotation are included in the metadata dictionary.

    Args:
        json_name (str): Path to the JSON file containing annotations.
        with_masks (bool): Whether to include masks in the Detections.
        classes (list[str]): List of class names.
        with_track_ids (bool): Whether to include tracking IDs.
        skip_unknown_classes (bool): Whether to skip unknown classes. Default is True.

    Returns:
        dict: Dictionary containing detection data, including metadata with properties.
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

    ## to add Detection.data three dict are created
    detection_data = {}
    xyxyxyxy_dict = {} ## for darwin rotated bounding boxes using ellipses
    properties_dict = {} ## for darwin properties like multi class

    tracker_id = None
    if with_track_ids:
        tracker_id = []

    for annotation in data["annotations"]:
        class_id = get_class_id(annotation, classes, skip_unknown_classes)
        if not isinstance(class_id, int):
            continue

        if "bounding_box" in annotation:
            xyxy_box, mask = add_box(annotation, with_masks, width, height)
        elif "ellipse" in annotation:
            xyxy_box, mask = add_ellipse(annotation, with_masks, width, height)
            ## add oriented bounding box
            xyxyxyxy = darwin_ellipse_to_xyxyxyxy(
                center=annotation["ellipse"]["center"],
                radius=annotation["ellipse"]["radius"],
                angle_rad=annotation["ellipse"]["angle"],
            )
            xyxyxyxy_dict[len(xyxy)] = xyxyxyxy
        else:
            ## TODO add keypoint
            continue  # skip annotations without bounding_box or ellipse

        # Include Darwin properties in Detections.data
        if annotation.get("properties", None) is not None:
            properties_dict[len(xyxy)] = annotation["properties"]

        # Include tracker Id in Detections.tracker_id
        if with_track_ids:
            tracker_id.append(annotation.get("instance_id", {}).get("value", None))

        xyxy.append(xyxy_box)
        class_ids.append(class_id)
        if with_masks:
            masks.append(mask)

    # To deal with empty darwin files
    if not xyxy:
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

    ## annotations are not always ellipse or do not have a property, still they need to be included!
    if xyxyxyxy_dict:
        detection_data["xyxyxyxy"] = []
        for i in range(len(xyxy)):
            detection_data["xyxyxyxy"].append(xyxyxyxy_dict.get(i, None))
    
    if properties_dict:
        detection_data["properties"] = []
        for i in range(len(xyxy)):
            detection_data["properties"].append(properties_dict.get(i, []))

    assert xyxy.shape[0] == len(class_ids), (
        f"xyxy len {xyxy.shape[0]}, but class_ids len {len(class_ids)}."
    )
    if with_masks:
        assert masks.shape[0] == len(class_ids), (
            f"masks len {len(masks)}, but class_ids len {len(class_ids)}."
        )
        return dict(xyxy=xyxy, class_id=class_ids, mask=masks, tracker_id=tracker_id, data=detection_data, metadata=metadata)
    else:
        return dict(xyxy=xyxy, class_id=class_ids, tracker_id=tracker_id, data=detection_data, metadata=metadata)


if __name__=="__main__":
    json_name = "/home/agro/w-drive-vision/GARdata/new_format/3710496261_broccoli_detection/datasets/anns/temporaryannottated/annotations/20170815_191925293_RGB.json"
    classes = ["healthy", "damaged", "mature", "cateye", "headrot", "broccoli"]
    temp = darwin_annotations_to_detections_dict(json_name, with_masks=True, classes=classes, with_radius=True, with_track_ids=True)

    print("a")