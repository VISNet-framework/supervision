import os
from pathlib import Path

import cv2
import numpy as np
import tqdm

from supervision.detection.core import Detections, reorder_detections
from supervision.detection.tools.transformers import (
    process_transformers_v5_panoptic_segmentation_result,
)
from supervision.detection.utils.converters import masks_to_semantic_mask
from supervision.utils.file import (
    find_valid_images_and_annotations,
    read_json_file,
    save_json_file,
)
from supervision.utils.image import advanced_crop_bbox


def semantic_mask_per_box(
    bgr_img: np.ndarray,
    annotation: Detections,
    mask: np.ndarray,
    image_name: Path,
    annotation_path: Path,
    images_directory_path: Path | None,
) -> list:
    """
    Generates COCO-style semantic segmentation for each bounding box in an image.

    Args:
        bgr_img (np.ndarray): The input image in BGR format.
        annotation (Detections): Detection results containing bounding boxes and IDs.
        mask (np.ndarray): Semantic mask corresponding to the input image.
        image_name (Path): Path object representing the image filename.
        annotation_path (Path): Path to the annotation file.
        images_directory_path (Path | None): Optional directory to save cropped images.

    Returns:
        list: list of COCO-style semantic segmentation for each bounding box.
    """
    coco_semseg_per_box = []
    image_list, mask_list, id_list = advanced_crop_bbox(
        bgr_img, annotation.xyxy, annotation.tracker_id, mask
    )

    for bgr_crop, mask_crop, unique_id in zip(image_list, mask_list, id_list):
        mask_name = annotation_path.parent / (image_name.stem + f"_{unique_id}.png")
        cv2.imwrite(str(mask_name), mask_crop.astype(np.uint8))

        ## you might want to save the cropped images as well
        new_image_name = image_name
        if images_directory_path is not None and bgr_img is not None:
            new_image_name = images_directory_path / mask_name.name
            cv2.imwrite(str(new_image_name), bgr_crop)

        coco_image = create_single_semseg(
            bgr_crop,
            new_image_name,
            mask_crop,
            mask_name,
            annotation_path,
        )
        coco_semseg_per_box.append(coco_image)
    return coco_semseg_per_box


def save_coco_semseg_annotations(
    dataset,
    images_directory_path: str,
    annotation_path: str,
    semseg_per_box: bool = False,
    segmentation_order: list[str] | None = None,
    skip_classes: list[str] | None = None,
) -> None:
    """
    Save semantic segmentation annotations in COCO semseg format for a given detection
    dataset.

    Args:
        dataset (DetectionDataset): The dataset containing images and annotations.
        images_directory_path: (str) Path to save images if semseg_per_box=True
        annotation_path (str): Path to save the resulting COCO-format annotation JSON
            file.
        semseg_per_box (bool, optional): If True, generate a separate mask per bounding
            box or object. Defaults to False.
        segmentation_order (list[str], optional): list of class names specifying the
            order of creating the mask. For example, if
            semgmentation_order=["leaf", "disease1", "disease2"], it will first create
            the mask of leaf, then overlay with disease.
        Might be useful in certain scenarios for example, if annotations are overlaying
            / not subtracted. Defaults to None.
        skip_classes (list[str], optional): list of class names to skip when generating
            masks. Defaults to None.

    Returns:
        None
    """
    # Detections and optionally save json.
    if images_directory_path is not None:
        images_directory_path = Path(images_directory_path)
        images_directory_path.mkdir(exist_ok=True)

    annotation_path = Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    # Example: handle segmentation_order being None or a list of strings
    if segmentation_order is None:
        segmentation_order_id = []
    else:
        segmentation_order_id = []
        for class_name in segmentation_order:
            if class_name in dataset.classes:
                segmentation_order_id.append(dataset.classes.index(class_name))
            else:
                raise ValueError(f"Class '{class_name}' not found in dataset.classes")

    # Create list with indexes which mask are not created
    if skip_classes is None:
        idx_skip_classes = []
    else:
        not_found = [cls for cls in skip_classes if cls not in dataset.classes]
        if not_found:
            raise ValueError(f"Classes {not_found} not found in dataset.classes")
        idx_skip_classes = [
            idx
            for idx, class_name in enumerate(dataset.classes)
            if class_name in skip_classes
        ]

    coco_semseg_annotations = []
    with tqdm.tqdm(dataset, desc="Creating coco_semseg images", unit="img") as pbar:
        for image_path, bgr_img, annotation in pbar:
            image_path = Path(image_path)

            image_path_absolute = image_path.resolve()
            image_path_rel = Path(
                os.path.relpath(image_path_absolute, start=annotation_path.parent)
            )

            # reorder annotations
            annotation = reorder_detections(annotation, segmentation_order)

            # create a mask
            mask = masks_to_semantic_mask(
                masks=annotation.mask,
                class_ids=annotation.class_id,
                idx_skip_classes=idx_skip_classes,
            )

            # if true a semantic mask will be created per box
            if semseg_per_box:
                coco_semseg_annotations.extend(
                    semantic_mask_per_box(
                        bgr_img,
                        annotation,
                        mask,
                        image_path_rel,
                        annotation_path,
                        images_directory_path,
                    )
                )
            else:  ## just create a normal mask
                mask_name = annotation_path.parent / (image_path.stem + ".png")
                cv2.imwrite(str(mask_name), mask.astype(np.uint8))

                coco_image = create_single_semseg(
                    bgr_img,
                    image_path,
                    mask,
                    mask_name,
                    annotation_path,
                )
                coco_semseg_annotations.append(coco_image)

    save_json_file(coco_semseg_annotations, file_path=annotation_path)


def create_single_semseg(
    bgr_img: np.ndarray,
    image_name: str | Path,
    mask: np.ndarray,
    mask_name: str | Path,
    annotation_path: Path,
) -> dict:
    """
    Create a COCO image dictionary for a single semantic segmentation mask.

    Args:
        bgr_img (np.ndarray): The image array (BGR format).
        image_name (str or Path): The image file name or path.
        mask (np.ndarray): The semantic segmentation mask.
        mask_name (str or Path): The mask file name or path.
        annotation_path (Path): Path to the annotation file (used for relative paths).

    Returns:
        dict: COCO image dictionary containing file names and image dimensions.
    """
    if len(bgr_img.shape) == 2:
        image_height, image_width = bgr_img.shape
    else:
        image_height, image_width = bgr_img.shape[:2]

    if len(mask.shape) == 2:
        mask_height, mask_width = mask.shape
    else:
        mask_height, mask_width = mask.shape[:2]

    if (image_height, image_width) != (mask_height, mask_width):
        raise ValueError(
            f"Image and mask dimensions do not match: "
            f"image ({image_height}, {image_width}), "
            f"mask ({mask_height}, {mask_width})"
        )

    coco_image = {
        "file_name": os.path.relpath(str(image_name), start=annotation_path.parent),
        "sem_seg_file_name": os.path.relpath(
            str(mask_name), start=annotation_path.parent
        ),
        "height": image_height,
        "width": image_width,
    }
    return coco_image


def load_from_semseg_dir(
    images_directory_path: str,
    annotations_path: str,
) -> tuple[list[str], dict[str, Detections]]:
    """
    Loads images and their corresponding semantic segmentation masks.

    Args:
        images_directory_path (str): Path to the directory containing image files.
        annotations_path (str): Path to the directory containing annotation (mask) files

    Returns:
        tuple[list[str], dict[str, Detections]]:
            A tuple containing a list of image file paths and a dictionary mapping image
            paths to Detections objects.
    """

    images_directory_path = Path(images_directory_path)
    annotation_directory_path = Path(annotations_path)
    images_paths, annotation_paths = find_valid_images_and_annotations(
        images_directory_path=images_directory_path,
        annotation_path=annotation_directory_path,
        annotation_extentions=["png"],
    )
    images = []
    annotations = {}

    with tqdm.tqdm(
        zip(images_paths, annotation_paths),
        total=len(images_paths),
        desc="Loading semantic annotations",
    ) as pbar:
        for img_name, annot_name in pbar:
            mask = cv2.imread(str(annot_name), -1)
            annotation = Detections(
                xyxy=np.array([[0, 0, mask.shape[1], mask.shape[0]]]),
                mask=mask[np.newaxis, ...],
            )
            images.append(str(img_name))
            annotations[str(img_name)] = annotation

    return images, annotations


def load_coco_semseg_annotations(
    images_directory_path: str | None,
    annotations_path: str,
    id2label: dict | None = None,
) -> tuple[list[str], dict[str, Detections]]:
    """
    Loads COCO-style semantic segmentation annotations and corresponding images.

    Args:
        images_directory_path (str | None): Optional path to the directory containing
            images. If provided, image paths are resolved relative to this directory;
            otherwise, they are resolved relative to the annotations file.
        annotations_path (str): Path to the COCO semantic segmentation annotations
            JSON file.
        id2label (dict | None, optional): Optional mapping from mask IDs to class label
            Used to decode segmentation masks into Detections objects.

    Returns:
        tuple[list[str], dict[str, Detections]]:
            - list of resolved image file paths.
            - Dictionary mapping image file paths to their corresponding
              Detections objects.

    Raises:
        FileNotFoundError: If an image file specified in the annotations is not found.
    """

    coco_semseg_data = read_json_file(file_path=annotations_path)

    images = []
    annotations = {}

    for coco_image in coco_semseg_data:
        image_name = coco_image["file_name"]
        mask_name = coco_image["sem_seg_file_name"]

        image_path = (Path(annotations_path).parent / image_name).resolve()
        if images_directory_path is not None:
            image_path = (images_directory_path / image_name).resolve()

        if not image_path.exists():
            print(f"Image file not found when loading coco_semseg: {image_path}")
            raise FileNotFoundError

        mask_path = str((Path(annotations_path).parent / mask_name).resolve())
        mask = cv2.imread(mask_path, -1)
        annotation = Detections(
            **process_transformers_v5_panoptic_segmentation_result(mask, id2label)
        )

        images.append(str(image_path))
        annotations[str(image_path)] = annotation

    return images, annotations
