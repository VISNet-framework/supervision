from pathlib import Path

import cv2
import numpy as np

from supervision import Detections
from supervision.dataset.formats import darwin as dataset_darwin
from supervision.detection.tools import darwin as detection_darwin


def create_image(create_image=False):
    image = np.zeros((640, 640), dtype=np.uint8)

    image_circle = cv2.circle(image, (320, 320), 250, 255, thickness=-1)

    image_circle_hole = cv2.circle(image_circle.copy(), (320, 320), 50, 0, thickness=-1)

    image_circle_holes = cv2.circle(
        image_circle_hole.copy(), (450, 450), 25, 0, thickness=-1
    )

    image_circle_holes_complex = image_circle_holes.copy()
    image_circle_holes_complex[:, 180:220] = 0
    image_circle_holes_complex[:, 400:410] = 0

    image_circle_holes_complex_double = image_circle_holes_complex.copy()
    image_circle_holes_complex_double = cv2.circle(
        image_circle_holes_complex_double.copy(), (320, 320), 25, 255, thickness=-1
    )

    if create_image:
        cv2.imwrite("circle.png", image)
        cv2.imwrite("circle_with_hole.png", image_circle_hole)
        cv2.imwrite("circle_with_holes.png", image_circle_holes)
        cv2.imwrite("circle_with_holes_complex_mask.png", image_circle_holes_complex)
        cv2.imwrite(
            "circle_holes_complex_double.png", image_circle_holes_complex_double
        )

    return {
        "circle": image_circle,
        "circle_with_hole": image_circle_hole,
        "circle_with_holes": image_circle_holes,
        "circle_with_holes_complex_mask": image_circle_holes_complex,
        "circle_holes_complex_double": image_circle_holes_complex_double,
    }


def check_mask_similarity(mask1, mask2, threshold=0.97):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    similarity = intersection / union if union != 0 else 0
    assert similarity >= threshold, f"Mask similarity {similarity:.3f}"


def check_similarity_darwin_and_mask(json_name="circle.json", img_dict={}):
    current_dir = Path(__file__).resolve().parent / "data"
    json_path = current_dir / json_name

    ## check conversion of darwin json to mask
    detections = Detections.from_darwin(
        json_name=json_path,
        with_masks=True,
        classes=["dumy"],
        with_ellipse_as="mask",
        with_track_ids=False,
        skip_unknown_classes=True,
    )
    assert np.max(detections.mask) == 1

    # real_mask = cv2.imread(str(img_path), -1)>0
    real_mask = img_dict[json_path.stem] > 0

    darwin_mask = detections.mask[0] > 0
    # no threshold of 1 because of small errors in manual annotations
    check_mask_similarity(real_mask, darwin_mask, threshold=0.975)

    # check if generated mask from conversion of mask2polyon and
    # poly2mask is same as ground truth
    polygon_dict = dataset_darwin._detection_mask_to_darwin_polygon(
        real_mask, 0.0, 1.0, 0.0
    )
    converted_mask = detection_darwin.darwin_polygon_to_mask(
        polygon_dict, height=real_mask.shape[0], width=real_mask.shape[1]
    )
    check_mask_similarity(real_mask, converted_mask > 0, threshold=1)


def test_darwin_image_circles():
    img_dict = create_image()
    check_similarity_darwin_and_mask(json_name="circle.json", img_dict=img_dict)
    check_similarity_darwin_and_mask(
        json_name="circle_with_hole.json", img_dict=img_dict
    )
    check_similarity_darwin_and_mask(
        json_name="circle_with_holes.json", img_dict=img_dict
    )
    check_similarity_darwin_and_mask(
        json_name="circle_with_holes_complex_mask.json", img_dict=img_dict
    )
    check_similarity_darwin_and_mask(
        json_name="circle_holes_complex_double.json", img_dict=img_dict
    )
