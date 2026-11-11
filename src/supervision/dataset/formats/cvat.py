from collections import defaultdict
from pathlib import Path

import defusedxml.ElementTree as ET
import numpy as np
import tqdm
from natsort import natsorted

from supervision.detection.compact_mask import CompactMask
from supervision.detection.core import Detections
from supervision.detection.utils.converters import mask_to_xyxy, polygons_to_mask
from supervision.utils.image import load_image_shape_quick


def load_cvat_video_annotations(
    images_directory_path: str | Path,
    annotation_path: str | Path,
    attribute_class: str = "class",
    attribute_class_backup: str = "species",
    additional_float_attributes: list[str] | None = None,
    include_empty=False,
) -> tuple[list[str], list[str], dict[str, Detections]]:
    """Load CVAT video annotations, merging polygons by object and frame.

    Args:
        images_directory_path: Directory containing the video frame images.
        annotation_path: Path to the CVAT video annotation XML file.
        attribute_class: Track attribute to use as the class name.
        attribute_class_backup: Track attribute to use as the class name
            (for example not all species in dataset have a FAO code, common
             or scientific name can be used instead).
        additional_float_attributes: Reserved for loading extra track attributes.

    Returns:
        A tuple containing sorted class names, ordered image paths, and a mapping
        from each image path to its detections. Polygons from separate CVAT tracks
        that share an object ID in a frame are merged into one instance mask.
    """
    if additional_float_attributes is not None:
        raise NotImplementedError("Please make this")

    images_directory_path = Path(images_directory_path).resolve()
    image_paths = natsorted(str(path) for path in images_directory_path.glob("*"))
    height, width, _ = load_image_shape_quick(image_paths[0])

    annotation_path = Path(annotation_path).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing {annotation_path}")
    ann_tree = ET.parse(annotation_path)
    ann_root = ann_tree.getroot()

    frame_polygons: defaultdict[int, defaultdict[int, list[tuple[str, int, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    classnames: set[str] = set()

    # TODO create mapping from object pids to object ids, since
    # in the optifish hirtshals data pid is a string, object id needs to be a int
    next_object_id = 0
    object_pid_to_id = dict()

    tracks = ann_root.findall("track")
    for track in tqdm.tqdm(tracks, colour="blue", unit="track"):
        track_id = int(track.attrib["id"])
        firstchoice_classname = track.find(f"attribute[@name='{attribute_class}']")
        if firstchoice_classname is not None:
            classname = firstchoice_classname.text
        else:
            secondchoice_classname = track.find(
                f"attribute[@name='{attribute_class_backup}']"
            )
            if secondchoice_classname is not None:
                classname = track.find(
                    f"attribute[@name='{attribute_class_backup}']"
                ).text
            else:
                print(
                    f"Unable to find classname for track id {track_id},"
                    "need to investigate!"
                )
                continue
        # object_id = int(track.find("attribute[@name='object_id']").text)
        object_pid = track.find("attribute[@name='object_id']").text
        object_pid_to_id[object_pid] = next_object_id
        next_object_id += 1
        classnames.add(classname)

        # Group XML polygons first so masks are rasterized only after frame/object
        # membership and each polygon's track metadata are known.
        for polygon in track.findall("polygon"):
            frame_id = int(polygon.attrib["frame"])
            frame_polygons[frame_id][object_pid].append(
                (polygon.attrib["points"], track_id, classname)
            )

    classnames_sorted = sorted(classnames)
    class_to_id = {
        classname: class_id for class_id, classname in enumerate(classnames_sorted)
    }
    annotations: dict[str, Detections] = {}

    for frame_id in tqdm.trange(len(image_paths), unit="frame"):
        image_path = image_paths[frame_id]
        object_polygons = frame_polygons[frame_id]
        if not object_polygons:
            if include_empty:
                annotations[image_path] = Detections.empty()
            continue

        compact_masks: list[CompactMask] = []
        xyxy_list: list[np.ndarray] = []
        class_ids: list[int] = []
        object_ids: list[int] = []

        for object_pid, polygons in object_polygons.items():
            track_ids = {track_id for _, track_id, _ in polygons}
            object_classnames = {classname for _, _, classname in polygons}
            if len(object_classnames) != 1:
                raise ValueError(
                    f"Object ID {object_pid} has conflicting class names in frame "
                    f"{frame_id} across CVAT tracks {sorted(track_ids)}: "
                    f"{sorted(object_classnames)}"
                )
            classname = next(iter(object_classnames))

            polygons_np = [
                np.fromstring(points_str.replace(";", ","), sep=",").reshape(-1, 2)
                for points_str, _, _ in polygons
            ]
            merged_mask = polygons_to_mask(
                polygons_np, resolution_wh=(width, height)
            ).astype(bool)

            object_xyxy = mask_to_xyxy(merged_mask[None, ...])
            compact_masks.append(
                CompactMask.from_dense(
                    merged_mask[None, ...],
                    xyxy=object_xyxy,
                    image_shape=(height, width),
                )
            )
            xyxy_list.append(object_xyxy[0])
            class_ids.append(class_to_id[classname])
            # The object ID is stable when CVAT splits one object across tracks.
            object_ids.append(object_pid_to_id[object_pid])

        annotations[image_path] = Detections(
            xyxy=np.asarray(xyxy_list),
            mask=CompactMask.merge(compact_masks),
            class_id=np.asarray(class_ids),
            tracker_id=np.asarray(object_ids),
        )

    if not include_empty:
        image_paths = [p for p in image_paths if p in annotations]
    return classnames_sorted, image_paths, annotations


def save_cvat_video_annotations() -> None:
    """Placeholder for the not-yet-implemented CVAT video exporter."""
    return
