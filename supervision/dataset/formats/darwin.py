import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

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


# def coco_categories_to_classes(coco_categories: List[dict]) -> List[str]:
#     return [
#         category["name"]
#         for category in sorted(coco_categories, key=lambda category: category["id"])
#     ]


# def build_darwin_class_index_mapping(
#     darwin_classes: List[str], target_classes: List[str]
# ) -> Dict[int, int]:
#     source_class_to_index = {class_name: idx for idx, class_name in enumerate(darwin_classes)}
#     return {
#         source_class_to_index[target_class_name]: target_class_index
#         for target_class_index, target_class_name in enumerate(target_classes)
#         if target_class_name in source_class_to_index
#     }


# def classes_to_coco_categories(classes: List[str]) -> List[dict]:
#     return [
#         {
#             "id": class_id,
#             "name": class_name,
#             "supercategory": "common-objects",
#         }
#         for class_id, class_name in enumerate(classes)
#     ]


# def group_coco_annotations_by_image_id(
#     coco_annotations: List[dict],
# ) -> Dict[int, List[dict]]:
#     annotations = {}
#     for annotation in coco_annotations:
#         image_id = annotation["image_id"]
#         if image_id not in annotations:
#             annotations[image_id] = []
#         annotations[image_id].append(annotation)
#     return annotations


# def coco_annotations_to_masks(
#     image_annotations: List[dict], resolution_wh: Tuple[int, int]
# ) -> npt.NDArray[np.bool_]:
#     return np.array(
#         [
#             rle_to_mask(
#                 rle=np.array(image_annotation["segmentation"]["counts"]),
#                 resolution_wh=resolution_wh,
#             )
#             if image_annotation["iscrowd"]
#             else polygon_to_mask(
#                 polygon=np.reshape(
#                     np.asarray(image_annotation["segmentation"], dtype=np.int32),
#                     (-1, 2),
#                 ),
#                 resolution_wh=resolution_wh,
#             )
#             for image_annotation in image_annotations
#         ],
#         dtype=bool,
#     )


def darwin_annotations_to_detections(
    json_name: str, with_masks: bool, class_name_2_index: dict,
) -> Detections:
    if not json_name:
        return Detections.empty()

    data = read_json_file(json_name)
    height = data["item"]["slots"][0]["height"]
    width = data["item"]["slots"][0]["width"]
    # class_ids = [
    #     image_annotation["category_id"] for image_annotation in image_annotations
    # ]

    # xyxy[:, 2:4] += xyxy[:, 0:2]

    # if with_masks:
    #     mask = coco_annotations_to_masks(
    #         image_annotations=image_annotations, resolution_wh=resolution_wh
    #     )
    #     return Detections(
    #         class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask
    #     )
    # else:
    xyxy, class_ids, masks = [], [], []
    for annotation in data["annotations"]:
        if "bounding_box" in annotation:
            class_id = class_name_2_index.get(annotation["name"], None)
            if class_id is None:
                print(f'skipping {annotation["name"]}')
                continue
            class_ids.append(class_id)

            xyxy.append([
                annotation["bounding_box"]["x"],
                annotation["bounding_box"]["y"],
                annotation["bounding_box"]["x"] + annotation["bounding_box"]["w"],
                annotation["bounding_box"]["y"] + annotation["bounding_box"]["h"]
            ])


            if with_masks:
                mask = np.zeros((height, width), dtype=bool)
                if "polygon" in annotation:
                    for path_points in annotation["polygon"]["paths"]:  # Darwin 2.0
                        points = []
                        for pp in range(len(path_points)):
                            points.append(path_points[pp]["x"])
                            points.append(path_points[pp]["y"])
                        points = np.array(points, dtype=np.int32).reshape(-1, 2)
                        mask = np.logical_or(mask, polygon_to_mask(points, (width, height))!=0)
                    masks.append(mask)
                else:
                    masks.append(mask)
    xyxy = np.asarray(xyxy)
    class_ids =  np.asarray(class_ids, dtype=int)
    assert xyxy.shape[0] == len(class_ids), f"Mismatch in shapes: xyxy has {xyxy.shape[0]} rows, but class_ids has {len(class_ids)} elements."
    if with_masks:
        masks = np.array(masks)
        assert masks.shape[0] == len(class_ids), f"Mismatch in shapes: masks has {len(masks)} elements, but class_ids has {len(class_ids)} elements."
    if with_masks and masks.sum()!=0:
        return Detections(xyxy=xyxy, class_id=class_ids, mask=masks)
    return Detections(xyxy=xyxy, class_id=class_ids)


# def detections_to_coco_annotations(
#     detections: Detections,
#     image_id: int,
#     annotation_id: int,
#     min_image_area_percentage: float = 0.0,
#     max_image_area_percentage: float = 1.0,
#     approximation_percentage: float = 0.75,
# ) -> Tuple[List[Dict], int]:
#     coco_annotations = []
#     for xyxy, mask, _, class_id, _, _ in detections:
#         box_width, box_height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
#         segmentation = []
#         iscrowd = 0
#         if mask is not None:
#             iscrowd = contains_holes(mask=mask) or contains_multiple_segments(mask=mask)

#             if iscrowd:
#                 segmentation = {
#                     "counts": mask_to_rle(mask=mask),
#                     "size": list(mask.shape[:2]),
#                 }
#             else:
#                 segmentation = [
#                     list(
#                         approximate_mask_with_polygons(
#                             mask=mask,
#                             min_image_area_percentage=min_image_area_percentage,
#                             max_image_area_percentage=max_image_area_percentage,
#                             approximation_percentage=approximation_percentage,
#                         )[0].flatten()
#                     )
#                 ]
#         coco_annotation = {
#             "id": annotation_id,
#             "image_id": image_id,
#             "category_id": int(class_id),
#             "bbox": [xyxy[0], xyxy[1], box_width, box_height],
#             "area": box_width * box_height,
#             "segmentation": segmentation,
#             "iscrowd": iscrowd,
#         }
#         coco_annotations.append(coco_annotation)
#         annotation_id += 1
#     return coco_annotations, annotation_id


# def read_darwin_json(
#     ann_filename: str, cfg: str, darwin_version: int = 2
# ): #-> list[Annotation]
#     """Reads darwin json file and returns annotations in mlutils format

#     Args:
#         ann_filename (str): annotation filename
#         cfg (Config): mlutils config object
#         darwin_version (int): darwin version (1 or 2)

#     Returns:
#         list[Annotation]: annotations in mlutils format
#     """
#     with open(ann_filename, "r") as json_file:
#         jsondata = json.load(json_file)

#     if darwin_version == 1:
#         h = jsondata["image"]["height"]
#         w = jsondata["image"]["width"]

#         img_name = jsondata["image"]["filename"]
#         darwin_folder = jsondata["image"]["path"]

#     if darwin_version == 2:
#         h = jsondata["item"]["slots"][0]["height"]
#         w = jsondata["item"]["slots"][0]["width"]

#         img_name = jsondata["item"]["name"]
#         darwin_folder = jsondata["item"]["path"]

#     annotations = []
#     path_names = ["", "path", "paths"]
#     keypoints_buffer = []

#     # if cfg.operation == "instance_segmentation":
#     #     if cfg.keypoint_enabled:
#     #         for p in jsondata["annotations"]:
#     #             if "keypoint" in p["name"]:
#     #                 if "nodes" in p["skeleton"]:
#     #                     nodes = p["skeleton"]["nodes"]
#     #                     keypoint_group = []
#     #                     for node in nodes:
#     #                         x = int(node["x"])
#     #                         y = int(node["y"])
#     #                         keypoint_name = node["name"]
#     #                         if y < h or x < w:
#     #                             visibility = 2
#     #                         else:
#     #                             visibility = 1
#     #                         id = cfg.keypoint_names.index(keypoint_name)
#     #                         keypoint_group.append(x)
#     #                         keypoint_group.append(y)
#     #                         keypoint_group.append(visibility)
#     #                     keypoints_buffer.append(keypoint_group)

#     for p in jsondata["annotations"]:
#         if with_masks:
#             mask = coco_annotations_to_masks(
#                 image_annotations=image_annotations, resolution_wh=resolution_wh
#             )
#             return Detections(
#                 class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask
#             )

#     return Detections(xyxy=xyxy, class_id=np.asarray(class_ids, dtype=int))
        # if cfg.single_class == 1:
        #     cl_name = cfg.class_names[0]
        # else:
        #     cl_name = p["name"]
        # if cl_name in cfg.class_names:
        #     a = Annotation()

        # Tasks:
        # object_detection
        # instance_segmentation
        # semantic_segmentation
        # if cfg.operation != "regression":
            # For object detection
            # if "bounding_box" in p:
            #     b = p["bounding_box"]
            #     bxy = [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
            #     a.box = [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
            #     a.box = list(map(int, a.box))[:4]
            #     a.area = b["w"] * b["h"]
            #     a.centroid = [int(bxy[0] + bxy[2]) / 2, int(bxy[1] + bxy[3]) / 2]
            #     a.cl = cfg.class_names.index(cl_name) + 1
            #     try:
            #         a.score = float(p["score"])
            #     except:
            #         a.score = 0
            #     a.iscrowd = 0

            #     # For instance/semantic segmentation
            #     a.polygons = []
            #     if darwin_version == 1:
            #         if "polygon" in p:
            #             pathx = path_names[darwin_version]
            #             if pathx in p["polygon"]:
            #                 points = []
            #                 path_points = p["polygon"]["path"]  # Darwin 2.0
            #                 for h in range(len(path_points)):
            #                     points.append(path_points[h]["x"])
            #                     points.append(path_points[h]["y"])
            #                 a.polygons.append(points)
            #         if "complex_polygon" in p:
            #             pathx = path_names[darwin_version]
            #             if pathx in p["complex_polygon"]:
            #                 for k in range(len(p["complex_polygon"][pathx])):
            #                     points = []
            #                     path_points = p["complex_polygon"]["path"][
            #                         k
            #                     ]  # Darwin 2.0
            #                     for pp in range(len(path_points)):
            #                         points.append(path_points[pp]["x"])
            #                         points.append(path_points[pp]["y"])
            #                     a.polygons.append(points)
            #     if darwin_version == 2:
            #         if "polygon" in p:
            #             pathx = path_names[darwin_version]

            #             if pathx in p["polygon"]:
            #                 for path_points in p["polygon"]["paths"]:  # Darwin 2.0
            #                     points = []
            #                     for pp in range(len(path_points)):
            #                         points.append(path_points[pp]["x"])
            #                         points.append(path_points[pp]["y"])
            #                     a.polygons.append(points)

            #     # For keypoints only
            #     if cfg.operation == "instance_segmentation":
            #         if cfg.keypoint_enabled:
            #             dx_vals = []

            #             # find keypoint with shortest distance from centroid
            #             for keypoint in keypoints_buffer:
            #                 mx = int(bxy[0] + bxy[2]) / 2
            #                 my = int(bxy[1] + bxy[3]) / 2
            #                 dx = distance.euclidean(
            #                     [mx, my], [keypoint[0], keypoint[1]]
            #                 )
            #                 dx_vals.append(dx)
            #             matching_keypoints = keypoints_buffer[
            #                 dx_vals.index(np.min(dx_vals))
            #             ]
            #             a.keypoint = matching_keypoints
            #             a.nkeypoint = len(cfg.keypoint_names)

            #     if cfg.operation != "object_detection":
            #         a.mask = iops.polygons2mask(a.polygons, h, w)
            #     annotations.append(a)

        # Tasks:
            # regression
            # classification
            # if cfg.operation == "regression" or cfg.operation == "image_classification":
            #     a.tag = p["name"]
            #     annotations.append(a)

    # img_info = {"img_name": img_name, "folder": darwin_folder, "imgsz": [h, w]}
    # return annotations, img_info



def load_darwin_annotations(
    images_directory_path: str,
    annotations_path: str,
    force_masks: bool = False,
    classes: list=[],
) -> Tuple[List[str], List[str], Dict[str, Detections]]:
    # coco_data = read_json_file(file_path=annotations_path)
    classes = classes #coco_categories_to_classes(coco_categories=coco_data["categories"])
    source_class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}

    # class_index_mapping = build_darwin_class_index_mapping(
    #     darwin_classes=coco_data["categories"], target_classes=classes
    # )
    # all_annotations = Path(annotations_path).rglob("*.json")
    import mlutils.dataset.dataset_utils as du

    image_annotation_pairs = du.find_valid_images_and_annotations(
        img_dir=images_directory_path,
        ann_dir=annotations_path,
        include_empty=True,
        n_samples="all",
    )

    # coco_images = coco_data["images"]
    # coco_annotations_groups = group_coco_annotations_by_image_id(
    #     coco_annotations=coco_data["annotations"]
    # )

    images = []
    annotations = {}

    for img_name, annot_name, _ in image_annotation_pairs:
        # image_name, image_width, image_height = (
        #     coco_image["file_name"],
        #     coco_image["width"],
        #     coco_image["height"],
        # )
        # image_annotations = coco_annotations_groups.get(coco_image["id"], [])
        # image_path = os.path.join(images_directory_path, image_name)
        annotation = darwin_annotations_to_detections(annot_name, force_masks, source_class_to_index)

        # annotation = coco_annotations_to_detections(
        #     image_annotations=image_annotations,
        #     resolution_wh=(image_width, image_height),
        #     with_masks=force_masks,
        # )
        # annotation = map_detections_class_id(
        #     source_to_target_mapping=class_index_mapping,
        #     detections=annotation,
        # )

        images.append(str(img_name))
        annotations[str(img_name)] = annotation

    return classes, images, annotations


# def save_coco_annotations(
#     dataset: "DetectionDataset",
#     annotation_path: str,
#     min_image_area_percentage: float = 0.0,
#     max_image_area_percentage: float = 1.0,
#     approximation_percentage: float = 0.75,
# ) -> None:
#     Path(annotation_path).parent.mkdir(parents=True, exist_ok=True)
#     licenses = [
#         {
#             "id": 1,
#             "url": "https://creativecommons.org/licenses/by/4.0/",
#             "name": "CC BY 4.0",
#         }
#     ]

#     coco_annotations = []
#     coco_images = []
#     coco_categories = classes_to_coco_categories(classes=dataset.classes)

#     image_id, annotation_id = 1, 1
#     for image_path, image, annotation in dataset:
#         image_height, image_width, _ = image.shape
#         image_name = f"{Path(image_path).stem}{Path(image_path).suffix}"
#         coco_image = {
#             "id": image_id,
#             "license": 1,
#             "file_name": image_name,
#             "height": image_height,
#             "width": image_width,
#             "date_captured": datetime.now().strftime("%m/%d/%Y,%H:%M:%S"),
#         }

#         coco_images.append(coco_image)
#         coco_annotation, annotation_id = detections_to_coco_annotations(
#             detections=annotation,
#             image_id=image_id,
#             annotation_id=annotation_id,
#             min_image_area_percentage=min_image_area_percentage,
#             max_image_area_percentage=max_image_area_percentage,
#             approximation_percentage=approximation_percentage,
#         )

#         coco_annotations.extend(coco_annotation)
#         image_id += 1

#     annotation_dict = {
#         "info": {},
#         "licenses": licenses,
#         "categories": coco_categories,
#         "images": coco_images,
#         "annotations": coco_annotations,
#     }
#     save_json_file(annotation_dict, file_path=annotation_path)
if __name__=="__main__":
    from pathlib import Path



    images_directory_path = "/home/agro/w-drive-vision/GARdata/datasets/3710496261_broccoli_detection/images/Paper04/"
    annotations_path = "/home/agro/w-drive-vision/GARdata/datasets/3710496261_broccoli_detection/anns/6-bbox-217images/annotations/"
    force_masks: bool = False,
    classes = ["healthy", "damaged", "mature", "cateye", "headrot"]  # (list[str]) target classes
    load_darwin_annotations(images_directory_path, annotations_path, force_masks=True,
                            classes=classes)