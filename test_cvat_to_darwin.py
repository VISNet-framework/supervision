import supervision as sv
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np
import argparse

from supervision.utils.file import find_valid_images_and_annotations


if __name__ == "__main__":
    # image_dir = Path("data_fishvis/data/images-5mp-rectified/rectified")
    # ann_dir = Path("data_fishvis/data/annotations-5mp-rectified/rectified")

    # for image_subdir in image_dir.glob("*batch_03-run_02*/"):
    #     print(f"{image_subdir=}")
    #     batchname = image_subdir.name
    #     ann_file = ann_dir / batchname / "annotations.xml"
    #     d = sv.DetectionDataset.from_cvat_video(
    #         str(image_subdir),
    #         str(ann_file),
    #         attribute_class="fao_code",
    #     )
    #     print(d.classes)
    #     d.as_darwin(
    #         f"fishvis_{batchname}",
    #         annotations_directory_path=f"data_fishvis/data/darwin/{batchname}",
    #         approximation_percentage=0.9
    #     )
    # exit()
    classes = [
        "BLL",
        "COD",
        "DAB",
        "FLE",
        "GUG",
        "HAD",
        "HER",
        "HKE",
        "LEM",
        "MON",
        "NEP",
        "PLA",
        "PLE",
        "POK",
        "SOL",
        "SQR",
        "WEG",
        "WHG",
    ]

    annotator_bbox = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    annotator_track = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
    annotator_label = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

    imgs, anns = find_valid_images_and_annotations(
        images_directory_path=Path(
            "data_fishvis/data/images-5mp-rectified/rectified/20231213-batch_03-run_02-high_occlusion"
        ),
        annotation_path=Path(
            "data_fishvis/data/darwin/20231213-batch_03-run_02-high_occlusion/"
        ),
    )
    for img_path, ann_path in tqdm(
        zip(imgs, anns, strict=True), total=len(imgs), colour="blue"
    ):
        tqdm.write(f"{img_path}")
        det = sv.Detections.from_darwin(
            ann_path, with_masks=True, classes=classes, with_track_ids=True
        )
        img = cv2.imread(str(img_path))
        img = annotator_bbox.annotate(img, det)
        img = annotator_track.annotate(img, det)
        temp_labels = np.array(classes)[det.class_id].tolist()
        if len(det) > 0:
            labels = [
                f"{classes[cid]:>5}: track {tid:<3}"
                for (cid, tid) in zip(det.class_id, det.tracker_id, strict=True)
            ]
            annotator_label.annotate(
                scene=img,
                detections=det,
                labels=labels,
            )

        cv2.imshow("Image", img)
        cv2.waitKey(0)
