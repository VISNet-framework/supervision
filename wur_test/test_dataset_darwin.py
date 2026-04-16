from pathlib import Path

import numpy as np
import pytest
from test_utils import mock_detections

from supervision.dataset.formats import darwin


@pytest.mark.parametrize(
    "xyxy, confidence, class_id, tracker_id, expected_bbox",
    [
        (
            [[10, 20, 30, 40]],
            [0.9],
            [0],
            [1],
            {"x": 10, "y": 20, "w": 20, "h": 20},
        ),
        (
            [[0, 0, 100, 100]],
            [1.0],
            [1],
            [2],
            {"x": 0, "y": 0, "w": 100, "h": 100},
        ),
    ],
)
def test_detections_to_darwin_dict_bbox(
    xyxy, confidence, class_id, tracker_id, expected_bbox
):
    detections = mock_detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        data={"properties": [{} for _ in class_id]},
    )
    image_shape = (100, 100)
    image_filename = Path("img.jpg")
    classes = ["cat", "dog"]
    darwin_dataset_name = "TestSet"
    result = darwin.detections_to_darwin_dict(
        detections=detections,
        image_shape=image_shape,
        image_filename=image_filename,
        classes=classes,
        darwin_dataset_name=darwin_dataset_name,
    )
    assert result["annotations"][0]["bounding_box"] == expected_bbox


@pytest.mark.parametrize(
    "xyxy, confidence, class_id, tracker_id",
    [
        ([[10, 20, 30, 40]], [0.9], [0], [1]),
    ],
)
def test__detection_xyxy_to_darwin_bbox_assertion(
    xyxy, confidence, class_id, tracker_id
):
    xyxy = np.array([1, 2, -1, 22])
    with pytest.raises(AssertionError):
        darwin._detection_xyxy_to_darwin_bbox(xyxy)


def test_find_valid_images_and_annotations(tmp_path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    (img_dir / "img1.jpg").write_bytes(b"fake")
    (img_dir / "img2.png").write_bytes(b"fake")
    (ann_dir / "img1.json").write_text("{}")
    (ann_dir / "img2.json").write_text("{}")
    images, annotations = darwin.find_valid_images_and_annotations(img_dir, ann_dir)
    assert len(images) == 2
    assert len(annotations) == 2
    assert all(isinstance(p, Path) for p in images)
    assert all(isinstance(p, Path) for p in annotations)


def test__detection_mask_to_darwin_polygon(monkeypatch):
    mask = np.zeros((10, 10), dtype=np.uint8)
    monkeypatch.setattr(
        darwin,
        "approximate_mask_with_polygons",
        lambda **kwargs: [np.array([[1, 2], [3, 4], [5, 6]])],
    )
    polygon = darwin._detection_mask_to_darwin_polygon(mask, 0.0, 1.0, 0.0)
    assert "paths" in polygon
    assert isinstance(polygon["paths"], list)
    assert polygon["paths"][0][0] == {"x": 1.0, "y": 2.0}
