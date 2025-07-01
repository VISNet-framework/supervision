from pathlib import Path

import numpy as np
import pytest
from test_utils import mock_detections

from supervision.dataset.formats import coco

dummy_mask = np.zeros([100,100], np.uint8)
dummy_mask[20:40, 10:30] = 1
dummy_mask2 = np.ones([100,100], np.uint8)

@pytest.mark.parametrize(
    "xyxy, confidence, class_id, tracker_id, mask",
    [
        (
            [[10, 20, 30, 40], [0,0,100,100]],
            [0.9,0.75],
            [2, 1],
            [1, 2],
            [dummy_mask, dummy_mask2],
        ),
    ],
)
def test_detections_to_coco_mask(
    xyxy, confidence, class_id, tracker_id, mask
):
    detections = mock_detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        mask=mask,
        data={"properties": [{} for _ in class_id]},
    )
    image_shape = (100, 100)
    image_filename = Path("img.jpg")
    classes = ["cat", "dog", "horse"]

    mask = coco.create_mask_coco_semseg(image_height=image_shape[0], image_width=image_shape[1], 
                                        temp_annotation=detections[0],
                                 idx_skip_classes=[],
                                 semseg_per_box=False)        
    assert mask.shape == image_shape
    assert mask.dtype == np.uint8
    assert mask.max() == 2

    ## test semseg per box
    mask = coco.create_mask_coco_semseg(image_height=image_shape[0], image_width=image_shape[1], 
                                        temp_annotation=detections[0],
                                 idx_skip_classes=[],
                                 semseg_per_box=True)        
    assert mask.shape == (20,20)
    assert mask.dtype == np.uint8
    assert mask.max() == 2
    assert mask.min() == 2 ## in specific example no zeros

    ## test skippping of classes
    mask = coco.create_mask_coco_semseg(image_height=image_shape[0], image_width=image_shape[1], 
                                        temp_annotation=detections[0],
                                 idx_skip_classes=[2],
                                 semseg_per_box=True)        
    assert mask.shape == (20,20)
    assert mask.dtype == np.uint8
    assert mask.max() == 0
    assert mask.min() == 0 ## in specific example no zeros