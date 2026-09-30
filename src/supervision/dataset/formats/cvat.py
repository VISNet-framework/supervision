from collections import defaultdict
from pathlib import Path

import defusedxml.ElementTree as ET
import numpy as np
import tqdm
from natsort import natsorted

from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.converters import mask_to_xyxy, polygon_to_mask
from supervision.utils.image import load_image_shape_quick


def load_cvat_video_annotations(
    images_directory_path: str | Path,
    annotation_path: str | Path,
    attribute_class: str = "class",
    additional_float_attributes: list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """
    Load cvat video annotations from image directory and .xml annotation file.

    attribute_class : the track attribute to use as the class name.
    # TODO add option to load other attributes as well
    """
    if additional_float_attributes is not None:
        raise NotImplementedError("Please make this")
    image_paths = []
    annotations = {}

    images_directory_path = Path(images_directory_path).resolve()
    # find image paths, we assume only images in directory
    image_paths = natsorted(str(f) for f in images_directory_path.glob("*"))
    # find image resolution (for video we assume the same resolution for every frame)
    (height, width, _) = load_image_shape_quick(image_paths[0])

    annotation_path = Path(annotation_path).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing {annotation_path}")
    ann_tree = ET.parse(annotation_path)
    ann_root = ann_tree.getroot()

    framedata = defaultdict(list)
    classnames = set()

    tracks = ann_root.findall("track")
    for track in tqdm.tqdm(tracks, colour="blue", unit="track"):
        track_id = int(track.attrib["id"])
        classname = track.find(f"attribute[@name='{attribute_class}']").text
        classnames.add(classname)
        # trackdata[track_id]["class"]
        # for attribute_name in additional_float_attributes:
        #     attribute_value = track.find(f"attribute[@name='{attribute_name}']").text
        #     additional_data[track_id][attribute_name] = float(attribute_value)

        # find all the image frames it is part of
        for poly in track.findall("polygon"):
            frame_id = int(poly.attrib["frame"])
            points_str = poly.attrib["points"]
            points = np.array(
                [
                    [float(p.split(",")[0]), float(p.split(",")[1])]
                    for p in points_str.split(";")
                ]
            )
            mask = polygon_to_mask(points, resolution_wh=(width, height))[None, ...]
            xyxy = mask_to_xyxy(masks=mask)
            mask_compact = CompactMask.from_dense(
                mask,
                xyxy=xyxy,
                image_shape=(height, width),
            )

            framedata[frame_id].append(
                {
                    "xyxy": xyxy,
                    "mask": mask_compact,
                    "track_id": track_id,
                    "classname": classname,
                }
            )

    classnames_sorted = sorted(classnames)
    for frame_id in range(len(image_paths)):
        image_path = image_paths[frame_id]
        if len(framedata[frame_id]) == 0:
            annotations[image_path] = Detections.empty()
        else:
            # masks = np.array([d["mask"] for d in framedata[frame_id]], dtype=bool)
            masks = CompactMask.merge([d["mask"] for d in framedata[frame_id]])
            class_ids = np.array(
                [classnames_sorted.index(d["classname"]) for d in framedata[frame_id]]
            )
            track_ids = np.array([d["track_id"] for d in framedata[frame_id]])
            xyxy = np.array([d["xyxy"][0] for d in framedata[frame_id]])

            annotations[image_path] = Detections(
                xyxy=xyxy,
                mask=masks,
                class_id=class_ids,
                tracker_id=track_ids,
            )
    return classnames_sorted, image_paths, annotations


def save_cvat_video_annotations():
    # not needed?
    return
