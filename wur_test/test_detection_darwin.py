import json

import numpy as np
import pytest

from supervision import Detections
from supervision.detection.tools import darwin

MIN_DARWIN_DICT = {
    "version": "2.0",
    "schema_ref": "<https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
    "item": {
        "name": "item_name.jpg",
        "path": "/",
        "slots": [{"type": "image", "slot_name": "0", "width": 160, "height": 100}],
    },
    "annotations": [],
}


def test_xyxyxyxy_to_xyxy():
    corners = [[1, 2], [5, 2], [5, 6], [1, 6]]
    result = darwin.xyxyxyxy_to_xyxy(corners)
    assert result == [1, 2, 5, 6]


def test_xyxyxyxy_to_xyxy_non_axis_aligned():
    corners = [[2, 3], [8, 1], [10, 7], [4, 9]]
    result = darwin.xyxyxyxy_to_xyxy(corners)
    assert result == [2, 1, 10, 9]


def test_darwin_ellipse_to_xyxyxyxy_and_xyxy():
    ellipse = {
        "center": {"x": 10, "y": 10},
        "radius": {"x": 5, "y": 3},
        "angle": 0.0,
    }
    corners = darwin.darwin_ellipse_to_xyxyxyxy(ellipse)
    assert len(corners) == 4
    for corner in corners:
        assert len(corner) == 2
    # No rotation, so corners should be axis-aligned
    np.testing.assert_allclose(corners, [[5, 7], [15, 7], [15, 13], [5, 13]])
    xyxy = darwin.darwin_ellipse_to_xyxy(ellipse)
    assert xyxy == [5, 7, 15, 13]


def test_darwin_ellipse_to_xyxyxyxy_rotation():
    ellipse = {
        "center": {"x": 0, "y": 0},
        "radius": {"x": 2, "y": 1},
        "angle": np.pi / 2,  # 90 degrees
    }
    corners = darwin.darwin_ellipse_to_xyxyxyxy(ellipse)
    # After 90 degree rotation, x and y axes swap
    np.testing.assert_allclose(
        np.round(corners, 2), [[1, -2], [1, 2], [-1, 2], [-1, -2]]
    )


def test_darwin_ellipse_to_mask():
    ellipse = {
        "center": {"x": 10, "y": 10},
        "radius": {"x": 5, "y": 3},
        "angle": 0.0,
    }
    mask = darwin.darwin_ellipse_to_mask(ellipse, height=20, width=20)
    assert mask.shape == (20, 20)
    assert mask.dtype == np.uint8
    assert np.max(mask) == 1
    assert np.min(mask) == 0


def test_darwin_ellipse_to_mask_partial_outside():
    ellipse = {
        "center": {"x": 0, "y": 0},
        "radius": {"x": 5, "y": 5},
        "angle": 0.0,
    }
    mask = darwin.darwin_ellipse_to_mask(ellipse, height=10, width=10)
    assert mask[0, 0] == 1  # Top-left corner should be inside ellipse


def test_empty_mask():
    mask = darwin.empty_mask(10, 15)
    assert mask.shape == (10, 15)
    assert np.all(mask == 0)


def test_darwin_polygon_to_mask():
    polygon = {
        "paths": [
            [{"x": 2, "y": 2}, {"x": 7, "y": 2}, {"x": 7, "y": 7}, {"x": 2, "y": 7}]
        ]
    }
    mask = darwin.darwin_polygon_to_mask(polygon, height=10, width=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert np.max(mask) == 1
    assert np.min(mask) == 0
    # The center should be inside the mask
    assert mask[4, 4] == 1


def test_darwin_polygon_to_mask_multiple_paths():
    polygon = {
        "paths": [
            [{"x": 1, "y": 1}, {"x": 3, "y": 1}, {"x": 3, "y": 3}, {"x": 1, "y": 3}],
            [{"x": 5, "y": 5}, {"x": 7, "y": 5}, {"x": 7, "y": 7}, {"x": 5, "y": 7}],
        ]
    }
    mask = darwin.darwin_polygon_to_mask(polygon, height=10, width=10)
    assert mask[2, 2] == 1
    assert mask[6, 6] == 1


def test_darwin_bounding_box_to_xyxy():
    bbox = {"x": 1, "y": 2, "w": 3, "h": 4}
    xyxy = darwin.darwin_bounding_box_to_xyxy(bbox)
    assert xyxy == [1, 2, 4, 6]


def test_darwin_bounding_box_to_xyxy_negative_coords():
    bbox = {"x": -5, "y": -5, "w": 10, "h": 10}
    xyxy = darwin.darwin_bounding_box_to_xyxy(bbox)
    assert xyxy == [-5, -5, 5, 5]


@pytest.mark.parametrize(
    "with_masks,with_ellipse_as,with_track_ids,skip_unknown_classes,metadata",
    [
        (True, None, True, True, {}),
        (False, None, False, True, {}),
        (True, "mask", False, False, {"meta": 1}),
        (False, "oriented_bounding_box", True, False, {}),
    ],
)
def test_from_darwin_empty(
    tmpdir, with_masks, with_ellipse_as, with_track_ids, skip_unknown_classes, metadata
):
    darwin_json_path = tmpdir / "darwin_dict.json"
    with open(darwin_json_path, "w") as f:
        json.dump(
            {
                "version": "2.0",
                "schema_ref": "dummy",
                "item": {
                    "name": "item.jpg",
                    "path": "/",
                    "slots": [
                        {"type": "image", "slot_name": "0", "width": 100, "height": 100}
                    ],
                },
                "annotations": [],
            },
            f,
        )

    detections = Detections.from_darwin(
        json_name=darwin_json_path,
        with_masks=with_masks,
        classes=["cat", "dog"],
        with_ellipse_as=with_ellipse_as,
        with_track_ids=with_track_ids,
        skip_unknown_classes=skip_unknown_classes,
        metadata=metadata,
    )
    assert detections.is_empty()
    assert detections.metadata == metadata
