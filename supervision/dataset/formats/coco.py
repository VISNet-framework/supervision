import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import tqdm

from supervision.dataset.utils import (
    approximate_mask_with_polygons,
    map_detections_class_id,
    mask_to_rle,
    rle_to_mask,
)
from supervision.detection.core import Detections
from supervision.detection.utils import (
    contains_holes,
    contains_multiple_segments,
    polygon_to_mask,
)
from supervision.utils.file import read_json_file, save_json_file

if TYPE_CHECKING:
    from supervision.dataset.core import DetectionDataset


def coco_categories_to_classes(coco_categories: List[dict]) -> List[str]:
    return [
        category["name"]
        for category in sorted(coco_categories, key=lambda category: category["id"])
    ]


def build_coco_class_index_mapping(
    coco_categories: List[dict], target_classes: List[str]
) -> Dict[int, int]:
    source_class_to_index = {
        category["name"]: category["id"] for category in coco_categories
    }
    return {
        source_class_to_index[target_class_name]: target_class_index
        for target_class_index, target_class_name in enumerate(target_classes)
    }


def classes_to_coco_categories(classes: List[str]) -> List[dict]:
    return [
        {
            "id": class_id,
            "name": class_name,
            "supercategory": "common-objects",
        }
        for class_id, class_name in enumerate(classes)
    ]


def group_coco_annotations_by_image_id(
    coco_annotations: List[dict],
) -> Dict[int, List[dict]]:
    annotations = {}
    for annotation in coco_annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(annotation)
    return annotations


def coco_annotations_to_masks(
    image_annotations: List[dict], resolution_wh: Tuple[int, int]
) -> npt.NDArray[np.bool_]:
    return np.array(
        [
            rle_to_mask(
                rle=np.array(image_annotation["segmentation"]["counts"]),
                resolution_wh=resolution_wh,
            )
            if image_annotation["iscrowd"]
            else polygon_to_mask(
                polygon=np.reshape(
                    np.asarray(image_annotation["segmentation"], dtype=np.int32),
                    (-1, 2),
                ),
                resolution_wh=resolution_wh,
            )
            for image_annotation in image_annotations
        ],
        dtype=bool,
    )


def coco_annotations_to_detections(
    image_annotations: List[dict], resolution_wh: Tuple[int, int], with_masks: bool
) -> Detections:
    if not image_annotations:
        return Detections.empty()

    class_ids = [
        image_annotation["category_id"] for image_annotation in image_annotations
    ]
    xyxy = [image_annotation["bbox"] for image_annotation in image_annotations]
    xyxy = np.asarray(xyxy)
    xyxy[:, 2:4] += xyxy[:, 0:2]

    if with_masks:
        mask = coco_annotations_to_masks(
            image_annotations=image_annotations, resolution_wh=resolution_wh
        )
        return Detections(
            class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask
        )

    return Detections(xyxy=xyxy, class_id=np.asarray(class_ids, dtype=int))


def detections_to_coco_annotations(
    detections: Detections,
    image_id: int,
    annotation_id: int,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> Tuple[List[Dict], int]:
    coco_annotations = []
    for xyxy, mask, _, class_id, _, _ in detections:
        box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        segmentation = []
        iscrowd = 0
        if mask is not None:
            iscrowd = contains_holes(mask=mask) or contains_multiple_segments(mask=mask)

            if iscrowd:
                segmentation = {
                    "counts": mask_to_rle(mask=mask),
                    "size": list(mask.shape[:2]),
                }
            else:
                segmentation = [
                    list(
                        approximate_mask_with_polygons(
                            mask=mask,
                            min_image_area_percentage=min_image_area_percentage,
                            max_image_area_percentage=max_image_area_percentage,
                            approximation_percentage=approximation_percentage,
                        )[0].flatten()
                    )
                ]
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [xyxy[0], xyxy[1], box_width, box_height],
            "area": box_width * box_height,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1
    return coco_annotations, annotation_id


def load_coco_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Detections]]:
    coco_data = read_json_file(file_path=annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_index_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
    coco_images = coco_data["images"]
    coco_annotations_groups = group_coco_annotations_by_image_id(
        coco_annotations=coco_data["annotations"]
    )

    images = []
    annotations = {}

    for coco_image in coco_images:
        image_name, image_width, image_height = (
            coco_image["file_name"],
            coco_image["width"],
            coco_image["height"],
        )
        image_annotations = coco_annotations_groups.get(coco_image["id"], [])
        image_path = str((Path(annotations_path).parent / image_name).resolve())

        annotation = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=force_masks,
        )
        annotation = map_detections_class_id(
            source_to_target_mapping=class_index_mapping,
            detections=annotation,
        )

        images.append(image_path)
        annotations[image_path] = annotation

    return classes, images, annotations


def save_coco_annotations(
    dataset: "DetectionDataset",
    annotation_path: str,
    min_image_area_percentage: float = 0.0,
    max_image_area_percentage: float = 1.0,
    approximation_percentage: float = 0.75,
) -> None:
    annotation_path = Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0",
        }
    ]

    coco_annotations = []
    coco_images = []
    coco_categories = classes_to_coco_categories(classes=dataset.classes)

    image_id, annotation_id = 1, 1
    for image_path, image, annotation in dataset:
        image_height, image_width, _ = image.shape
        # NOTE: we save the image name as a relative path
        # from the annotation file location
        image_path_absolute = Path(image_path).resolve()
        image_path_relative = os.path.relpath(
            image_path_absolute, start=annotation_path.parent
        )
        image_name = str(image_path_relative)
        coco_image = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
        }

        coco_images.append(coco_image)
        coco_annotation, annotation_id = detections_to_coco_annotations(
            detections=annotation,
            image_id=image_id,
            annotation_id=annotation_id,
            min_image_area_percentage=min_image_area_percentage,
            max_image_area_percentage=max_image_area_percentage,
            approximation_percentage=approximation_percentage,
        )

        coco_annotations.extend(coco_annotation)
        image_id += 1

    annotation_dict = {
        "info": {},
        "licenses": licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    save_json_file(annotation_dict, file_path=annotation_path)


def create_mask_coco_semseg_per_box(
    image_height: int,
    image_width: int,
    annotation: "Detections",
    annotation_path: Path,
    image_name: str,
    idx_skip_classes: List[int],
    bgr_img: np.ndarray | None = None,
    images_directory_path: Path | None = None,
) -> List[dict]:
    """
    Creates and saves semantic segmentation masks in COCO format for each object
    (bounding box) in the given annotation.

    Depending on the presence of unique tracker IDs, the function either creates a mask
    per annotation or per unique object (with the same tracker ID). Each mask is saved
    as a PNG file, and a corresponding COCO image dictionary is generated.

    Args:
        image_height (int): Height of the image.
        image_width (int): Width of the image.
        annotation (Detections): Annotation data containing object
            information and tracker IDs.
        annotation_path (Path): Path to the annotation file (used to determine where
            to save masks).
        image_name (str): Name of the image file.
        idx_skip_classes (List[int]): List of class indices to skip when creating masks.
        bgr_img (np.ndarray): numpy image
        images_directory_path (str): folder to save cropped images, if not specified
            images are not saved
    Returns:
        List[dict]: List of COCO image dictionaries, each containing file names and
            image dimensions.
    """

    def write_semseg(temp_annotation: Detections, unique_id: int):
        """
        Write coco segmentation based on unique_id
        return xyxy position
        """
        mask, xyxy = create_mask_coco_semseg(
            image_height,
            image_width,
            temp_annotation,
            idx_skip_classes,
            semseg_per_box=True,
        )
        mask_name = annotation_path.parent / (
            Path(image_name).stem + f"_{unique_id}.png"
        )

        if images_directory_path is not None and bgr_img is not None:
            new_image_name = images_directory_path / mask_name.name
            cv2.imwrite(
                str(new_image_name), bgr_img[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
            )
        else:
            new_image_name = image_name
        image_path_relative = os.path.relpath(
            new_image_name, start=annotation_path.parent
        )
        new_image_name = str(image_path_relative)

        cv2.imwrite(str(mask_name), mask.astype(np.uint8))
        coco_image = {
            "file_name": str(new_image_name),
            "semseg_file_name": os.path.relpath(
                str(mask_name), start=annotation_path.parent
            ),
            "height": image_height,
            "width": image_width,
        }
        return coco_image

    coco_semseg_annotations = []
    if None in annotation.tracker_id:
        unique_id = 0
        for temp_annotation in annotation:
            coco_image = write_semseg(temp_annotation, unique_id)
            coco_semseg_annotations.append(coco_image)
            unique_id += 1
    else:
        ## create object for every unique tracker id
        unique_tracker_id = np.unique(annotation.tracker_id)
        for unique_id in unique_tracker_id:
            temp_annotation = annotation[annotation.tracker_id == unique_id]
            coco_image = write_semseg(temp_annotation, unique_id)
            coco_semseg_annotations.append(coco_image)
    return coco_semseg_annotations


def create_mask_coco_semseg(
    image_height: int,
    image_width: int,
    temp_annotation: Detections,
    idx_skip_classes: list = [],
    semseg_per_box: bool = False,
):
    """
    Creates a semantic segmentation mask from sv.Detections annotations.
    Args:
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        temp_annotation (Detections): annotations containing masks and class IDs.
        idx_skip_classes (list, optional): List of class IDs to skip. Defaults to [].
        semseg_per_box (bool, optional): If True, returns mask cropped to the bounding
            box region. Defaults to False.
    Returns:
        np.ndarray: Semantic segmentation mask as a uint8 numpy array.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # temp_annotation.mask is N x image_height x image_width array with zero
    # or ones, therefore extract class_id
    for idx, instance_mask in enumerate(temp_annotation.mask):
        class_id = temp_annotation.class_id[idx]
        ## skip classes if exist
        if class_id in idx_skip_classes:
            continue
        mask = np.maximum(mask, (instance_mask.astype(np.uint8) * (class_id)))

    if semseg_per_box:
        # temp = np.argwhere(mask)
        # y1, x1 = temp.max(0)
        # y0, x0 = temp.min(0)

        temp_annotation.xyxy = temp_annotation.xyxy.astype(np.int64)

        ## mask_bounding based on min_max xyxy values
        x0 = np.clip(temp_annotation.xyxy[:, 0].min(), 0, image_width)
        y0 = np.clip(temp_annotation.xyxy[:, 1].min(), 0, image_height)
        x1 = np.clip(temp_annotation.xyxy[:, 2].max(), 0, image_width)
        y1 = np.clip(temp_annotation.xyxy[:, 3].max(), 0, image_height)

        return mask[y0:y1, x0:x1].astype(np.uint8), np.array([x0, y0, x1, y1])
    return mask.astype(np.uint8)


def reorder_annotation(annotation: Detections, segmentation_order_id: list[int]):
    """
    Reorders a Detections annotation so that objects with class IDs in
    segmentation_order_id appear first, in the given order. Remaining
    objects are appended in their original order.
    """
    order = []
    for class_id in segmentation_order_id:
        idxs = np.where(annotation.class_id == class_id)[0]
        order.extend(idxs.tolist())
    # Add any remaining indices not in segmentation_order_id
    remaining = [i for i in range(len(annotation.class_id)) if i not in order]
    order.extend(remaining)
    annotation = annotation[order]
    return annotation


def save_coco_semseg_annotations(
    dataset: "DetectionDataset",
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
        segmentation_order (list[str], optional): List of class names specifying the
            order of creating the mask. For example, if
            semgmentation_order=["leaf", "disease1", "disease2"], it will first create
            the mask of leaf, then overlay with disease.
        Might be useful in certain scenarios for example, if annotations are overlaying
            / not subtracted. Defaults to None.
        skip_classes (list[str], optional): List of class names to skip when generating
            masks. Defaults to None.

    Returns:
        None
    """
    # TODO future work, instead of using save_coco_semseg make a function to crop
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

    # Create list with indexes to skip
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

            if segmentation_order_id:
                # reorder annotations
                annotation = reorder_annotation(annotation, segmentation_order)

            image_height, image_width, _ = bgr_img.shape
            image_path_absolute = image_path.resolve()
            image_path_relative = os.path.relpath(
                image_path_absolute, start=annotation_path.parent
            )
            image_name = str(image_path_relative)

            # if true a semantic mask will be created per box
            if semseg_per_box:
                coco_semseg_annotations.extend(
                    create_mask_coco_semseg_per_box(
                        image_height=image_height,
                        image_width=image_width,
                        annotation=annotation,
                        annotation_path=annotation_path,
                        image_name=image_name,
                        idx_skip_classes=idx_skip_classes,
                        bgr_img=bgr_img,
                        images_directory_path=images_directory_path,
                    )
                )
            else:
                mask = create_mask_coco_semseg(
                    image_height, image_width, annotation, idx_skip_classes
                )

                mask_name = annotation_path.parent / (image_path.stem + ".png")
                cv2.imwrite(str(mask_name), mask.astype(np.uint8))

                coco_image = {
                    "file_name": str(image_name),
                    "semseg_file_name": os.path.relpath(
                        str(mask_name), start=annotation_path.parent
                    ),
                    "height": image_height,
                    "width": image_width,
                }
                coco_semseg_annotations.append(coco_image)

    save_json_file(coco_semseg_annotations, file_path=annotation_path)


if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path("/mnt/GARdata/datasets/project_name")
    annotations_path = base_dir / "anns/ann_version/coco/train.json"
    images_directory_path = base_dir / "images"
    force_masks: bool = (False,)
    classes = [
        "healthy",
    ]  # (list[str]) target classes
    load_coco_annotations(str(images_directory_path), str(annotations_path), True)
