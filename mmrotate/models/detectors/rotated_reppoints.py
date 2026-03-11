# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from .single_stage_msi import RotatedSingleStageDetectorMSI


@ROTATED_DETECTORS.register_module()
class RotatedRepPoints(RotatedSingleStageDetector):
    """Implementation of Rotated RepPoints."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RotatedRepPoints, self).__init__(backbone, neck, bbox_head,
                                               train_cfg, test_cfg, pretrained)


@ROTATED_DETECTORS.register_module()
class RotatedRepPointsMSI(RotatedSingleStageDetectorMSI):
    """MSI/DrMOD variant of Rotated RepPoints for OSSDET configs."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RotatedRepPointsMSI, self).__init__(backbone, neck, bbox_head,
                                                  train_cfg, test_cfg, pretrained)
