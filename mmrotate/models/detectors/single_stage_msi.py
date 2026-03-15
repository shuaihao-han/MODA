# Copyright (c) OpenMMLab. All rights reserved.
#
# MSI/DrMOD single-stage detector variant used by local experiments.
# This is intentionally NOT registered into the global detector registry to
# avoid clobbering the standard `RotatedSingleStageDetector` used by other
# models (e.g. RetinaNet). It is consumed only via inheritance by
# `RotatedRepPoints` so checkpoints that include CFB/SSAF/FMG/fem/extra_conv
# weights can be evaluated.

import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from mmrotate.core import rbbox2result
from ..builder import build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from tools.my_module.ssaf import SpectralSpatialAdaptiveFusion
from tools.my_module.cfb import CascadeFusionBlock
from tools.my_module.fem import FeatureEnhancementModule


class ForegroundMaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.DC_module = nn.ModuleList([
            ConvModule(
                in_channels=256,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
            ),
            ConvModule(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ),
        ])

    def forward(self, x):
        for layer in self.DC_module:
            x = layer(x)
        return torch.sigmoid(x)


class ActivationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-8

    def forward(self, fore_mask, gt_mask):
        hard_gt_mask = (gt_mask > 0).float()
        intersection = (fore_mask * gt_mask).sum()
        fore_sum = fore_mask.sum()
        gt_sum = gt_mask.sum()
        intersection_loss = 1 - (intersection + self.smooth) / (gt_sum + self.smooth)
        difference_loss = ((fore_mask * (1 - hard_gt_mask)).sum() + self.smooth) / (fore_sum + self.smooth)
        loss = intersection_loss + 0.1 * difference_loss
        return 0.6 * loss


def _create_gaussian_kernel(kernel_size, sigma, device):
    ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _apply_gaussian_smoothing(gt_mask, sigma=2.0):
    kernel_size = int(2 * (sigma * 3) + 1)
    kernel = _create_gaussian_kernel(kernel_size, sigma, gt_mask.device)
    kernel = kernel.expand(1, 1, -1, -1)
    padding = kernel_size // 2
    smoothed = F.conv2d(gt_mask, kernel, padding=padding)
    return gt_mask * 0.5 + smoothed


def generate_gt_mask_batch(img_shape, gt_bboxes, scale_factor=1 / 4):
    """Generate per-image GT masks (used only during training)."""
    B, _, H, W = img_shape
    mask_H = int(H * scale_factor)
    mask_W = int(W * scale_factor)
    masks = np.zeros((B, mask_H, mask_W), dtype=np.uint8)
    for i, gt_bbox in enumerate(gt_bboxes):
        for bbox in gt_bbox:
            x, y, w, h, theta = bbox
            rect = ((x * scale_factor, y * scale_factor),
                    (w * scale_factor + 2, h * scale_factor + 2),
                    np.degrees(theta))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(masks[i], [box], 1)
    masks = torch.from_numpy(masks).float()
    return _apply_gaussian_smoothing(masks.unsqueeze(1), sigma=1.0)


class RotatedSingleStageDetectorMSI(RotatedBaseDetector):
    """Single-stage detector variant that matches the MSI/DrMOD checkpoints."""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if pretrained:
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead'
            )
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Extra MSI modules (present in the checkpoint).
        self.CFB = CascadeFusionBlock(256)
        self.FMG = ForegroundMaskGenerator()
        self.ActLoss = ActivationLoss()
        self.SSAF = nn.ModuleList([SpectralSpatialAdaptiveFusion(256) for _ in range(3)])
        self.extra_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
        ])
        self.fem = nn.ModuleList([FeatureEnhancementModule(256) for _ in range(3)])
        self.conv_smooth = ConvModule(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
        )

        # This experiment deletes the unused fpn_convs (hence no weights in ckpt).
        if hasattr(self, 'neck') and hasattr(self.neck, 'fpn_convs'):
            del self.neck.fpn_convs

    @auto_fp16()
    def forward_neck(self, inputs):
        """A modified neck forward that returns laterals only."""
        assert len(inputs) == len(self.neck.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.neck.start_level])
            for i, lateral_conv in enumerate(self.neck.lateral_convs)
        ]
        laterals[-1] = self.CFB(laterals[-1], laterals[-1])
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.SSAF[i - 1](laterals[i], laterals[i - 1])
        return laterals

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.forward_neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        super().forward_train(img, img_metas)
        x = self.extract_feat(img)
        # GT mask supervision for FMG (training only)
        gt_masks = generate_gt_mask_batch(img.shape, gt_bboxes, scale_factor=1 / 4).to(img.device)
        fore_mask = self.FMG(x[0])
        x[0] = self.conv_smooth(x[0] * fore_mask + x[0])
        act_loss = self.ActLoss(fore_mask, gt_masks)

        for i in range(len(x) - 1):
            x[i + 1] = self.fem[i](x[i], x[i + 1])
        x.append(self.extra_conv[0](x[-1]))
        x.append(self.extra_conv[1](x[-1]))
        x = tuple(x[1:])

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses['act_loss'] = act_loss
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        fore_mask = self.FMG(x[0])
        x[0] = self.conv_smooth(x[0] * fore_mask + x[0])
        for i in range(len(x) - 1):
            x[i + 1] = self.fem[i](x[i], x[i + 1])
        x.append(self.extra_conv[0](x[-1]))
        x.append(self.extra_conv[1](x[-1]))

        x = tuple(x[1:])

        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        return [rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Fallback test-time augmentation handler.

        This MSI model is typically evaluated without TTA. To satisfy the
        BaseDetector interface, we run the first augmentation only.
        """
        return self.simple_test(imgs[0], img_metas[0], rescale=rescale)
