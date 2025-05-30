from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from natsort import natsorted

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.utils.file import (
    list_files_with_extensions_recursively,
    read_txt_file,
    save_text_file,
)

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset

"""
DOTAV2 format for oriented bounding boxes
"""


def detections_to_dota_annotations(
    detections: Detections,
    classes: List[str],
) -> list[str]:
    """
    Convert detections to DOTAV2 annotations format.

    Args:
        detections (Detections): Detections object containing detection data.
        classes (List[str]): List of class names.
    Returns:
        str: lines of DOTAV2 format to write to .txt file.
    """
    # Convert detections to DOTAV2 format
    lines = []
    for xyxy, _, _, class_id, data, _ in detections:
        assert ORIENTED_BOX_COORDINATES in data
        xyxyxyxy = data[ORIENTED_BOX_COORDINATES].flatten().tolist()
        assert len(xyxyxyxy) == 8, f"Expected 8 coordinates, got {len(xyxyxyxy)}"
        category = classes[class_id]
        line = [str(x) for x in xyxyxyxy]
        line.append(str(category))
        line.append(str(0))
        line = " ".join(line)
        lines.append(line)
    return lines


def save_dota_annotations(
    dataset: "DetectionDataset",
    annotations_directory_path: Path,
) -> None:
    """
    Save detections to DOTAV2 annotations format.

    Args:
        detections (Detections): Detections object containing detection data.
        image_path (Path): Path to the image file.
        classes (List[str]): List of class names.
        output_dir (Path): Directory to save the annotations.
    """
    # Create output directory if it doesn't exist
    annotations_directory_path.mkdir(parents=True, exist_ok=True)

    for image_path, image, annotation in dataset:
        image_path = Path(image_path)
        image_filename = image_path.name
        annotation_path = (annotations_directory_path / image_filename).with_suffix(
            ".txt"
        )
        lines = detections_to_dota_annotations(
            detections=annotation,
            classes=dataset.classes,
        )
        save_text_file(
            file_path=annotation_path,
            lines=lines,
        )
    return


def dota_annotation_to_detections(
    annotation_path: Path,
    classes: list[str],
) -> Detections:
    """
    Load DOTAV2 annotations from a .txt file.

    Args:
        image_path (Path): Path to the image file.
        annotations_directory_path (Path): Directory containing the annotations.

    Returns:
        List[Dict]: List of dictionaries containing annotation data.
    """
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    xyxys = []
    class_ids = []
    data = {ORIENTED_BOX_COORDINATES: []}

    lines = read_txt_file(annotation_path)
    for line in lines:
        parts = line.split(" ")
        assert len(parts) == 10
        (
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
        ) = [float(e) for e in parts[:8]]
        classname = parts[8]
        assert classname in classes, f"{classname} not in {classes}"
        xyxyxyxy = [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4],
        ]
        data[ORIENTED_BOX_COORDINATES].append(xyxyxyxy)
        xyxy = [
            min(x1, x2, x3, x4),
            min(y1, y2, y3, y4),
            max(x1, x2, x3, x4),
            max(y1, y2, y3, y4),
        ]
        xyxys.append(xyxy)
        class_ids.append(classes.index(classname))

    xyxys = np.array(xyxys, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    data[ORIENTED_BOX_COORDINATES] = np.array(
        data[ORIENTED_BOX_COORDINATES], dtype=np.float32
    )
    return Detections(xyxy=xyxys, class_id=class_ids, data=data)


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
        extensions=["txt"],
    )
    annotation_paths = natsorted(annotation_paths)

    image_paths = []
    for annotation_path in annotation_paths:
        # find the corresponding image path
        image_stem = annotation_path.stem
        image_path = image_candidate_paths[image_candidate_stems.index(image_stem)]
        image_paths.append(image_path)
    return image_paths, annotation_path


def load_dotav2_annotations(
    image_directory_path: Path,
    annotations_directory_path: Path,
    classes: list[str],
) -> None:
    image_paths, annotation_paths = find_valid_images_and_annotations(
        images_directory_path=image_directory_path,
        annotation_path=annotations_directory_path,
    )
    images = []
    annotations = {}
    for image_path, annotation_path in zip(image_paths, annotation_paths):
        image_path = str(image_path)
        images.append(image_path)
        annotation = dota_annotation_to_detections(annotation_path)
        annotations[image_path] = annotation
    return classes, images, annotations
