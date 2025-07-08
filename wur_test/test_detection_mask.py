import numpy as np
import pytest
from test_utils import mock_detections

from supervision.detection.core import reorder_detections
from supervision.detection.utils import masks_to_semantic_mask
from supervision.utils.image import advanced_crop_bbox

dummy_mask = np.zeros([100, 100], np.uint8)
dummy_mask[20:40, 10:30] = 1
dummy_mask2 = np.ones([100, 100], np.uint8)


@pytest.mark.parametrize(
    "xyxy, confidence, class_id, tracker_id, mask",
    [
        (
            [[10, 20, 30, 40], [10, 20, 30, 40], [0, 0, 100, 100]],
            [0.9, 0.75, 0.75],
            [2, 2, 1],
            [1, 1, 2],
            [dummy_mask, dummy_mask, dummy_mask2],
        ),
    ],
)
def test_detections_to_mask(xyxy, confidence, class_id, tracker_id, mask):
    detections = mock_detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        mask=mask,
        data={"properties": [{} for _ in class_id]},
    )
    image_shape = (100, 100)
    bgr_img = np.zeros((3, image_shape[0], image_shape[1]))

    ##  maximum is to because object is not nicely ordered
    mask = masks_to_semantic_mask(
        detections.mask,
        detections.class_id,
        idx_skip_classes=[],
    )

    assert mask.shape == image_shape
    assert mask.dtype == np.uint8
    assert mask.max() == 1  ## because overwritten by latest mask

    ##  after reordering detections mask.max() should contain 2 of first box
    detections = reorder_detections(detections, segmentation_order_id=[1, 2])
    assert np.all(detections.class_id == np.array([1, 2, 2]))

    mask = masks_to_semantic_mask(
        detections.mask,
        detections.class_id,
        idx_skip_classes=[],
    )
    assert mask.max() == 2

    detections = reorder_detections(detections, segmentation_order_id=[2, 1])

    image_list, mask_list, id_list = advanced_crop_bbox(
        bgr_img, detections.xyxy, None, mask
    )
    assert len(image_list) == 3
    assert mask_list[0].shape == (20, 20)
    assert mask_list[0].dtype == np.uint8
    assert mask_list[0].max() == 2
    assert mask_list[0].min() == 2

    ## now using tracking id
    image_list, mask_list, id_list = advanced_crop_bbox(
        bgr_img, detections.xyxy, detections.tracker_id, mask
    )
    assert len(image_list) == 2
    assert mask_list[0].shape == (20, 20)
    assert mask_list[0].dtype == np.uint8
    assert mask_list[0].max() == 2
    assert mask_list[0].min() == 2

    ## in specific example no zeros
    mask = masks_to_semantic_mask(
        detections.mask,
        detections.class_id,
        idx_skip_classes=[2],
    )

    image_list, mask_list, id_list = advanced_crop_bbox(
        bgr_img, detections.xyxy, detections.tracker_id, mask
    )

    assert len(image_list) == 2
    assert mask_list[0].shape == (20, 20)
    assert mask_list[0].dtype == np.uint8
    assert mask_list[1].max() == 1
    assert mask_list[1].min() == 1
