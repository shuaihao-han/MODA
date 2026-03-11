# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from mmcv.runner import BaseModule, auto_fp16
import torch.nn as nn
from tools.my_module.fem import FeatureEnhancementModule
from tools.my_module.ssaf import SpectralSpatialAdaptiveFusion
from tools.my_module.cfb import CascadeFusionBlock
from mmcv.cnn import ConvModule


# 前景掩码生成模块
class ForegroundMaskGenerator(nn.Module):
    def __init__(self):
        super(ForegroundMaskGenerator, self).__init__()
        self.DC_module = nn.ModuleList([
            # ConvModule(
            #     in_channels=256,
            #     out_channels=256,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     norm_cfg=dict(type='BN'),
            #     act_cfg=dict(type='ReLU')
            # ),
            ConvModule(
                in_channels=256,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=None,
                act_cfg=None
            )
        ])
        # 定义1个7x7的depth_wise卷积 + 1x1的point_wise卷积将通道维度降为1
        # self.DC_conv = DepthwiseSeparableConvModule(
        #     in_channels = 256,
        #     out_channels = 1, 
        #     kernel_size = 7, 
        #     stride = 1,
        #     padding = 3, 
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU')
        # )

    def forward(self, x):
        # 输入 x 的形状为 (B, C, H, W)
        for layer in self.DC_module:
            x = layer(x)
        # x = self.DC_conv(x)
        x = torch.sigmoid(x)  # 使用sigmoid将for_mask归一化到0,1
        # 不进行硬二值化，保持 x 为浮点数，以保持梯度信息
        return x  # 返回 x，而不是二值化后的张量


class ActivationLoss(nn.Module):
    def __init__(self):
        super(ActivationLoss, self).__init__()
        self.smooth = 1e-8
        # self.balance_factor = 0.8  # 用于平衡前景和背景损失

    def forward(self, fore_mask, gt_mask):
        # 确保 fore_mask 和 gt_mask 都是浮点数
        # gt_mask = gt_mask.float()
        hard_gt_mask = (gt_mask > 0).float()  # 生成hard_mask
        
        # 计算交集
        intersection = (fore_mask * gt_mask).sum()
        
        # 计算 fore_mask 和 gt_mask 的总和
        fore_sum = fore_mask.sum()
        gt_sum = gt_mask.sum()

        intersection_loss = 1 - (intersection + self.smooth) / (gt_sum + self.smooth)
        # 计算差集损失
        difference_loss = ((fore_mask * (1 - hard_gt_mask)).sum() + self.smooth) / (fore_sum + self.smooth)
        # 总损失为交集损失和差集损失之和
        loss = intersection_loss + 0.1 * difference_loss  # 将前景与背景的关注比设置为10 : 1

        return 0.4 * loss  # (调整act_loss所占的比例)
    
# 创建高斯核
def create_gaussian_kernel(kernel_size, sigma):
    ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    xx, yy = torch.meshgrid([ax, ax])
    kernel = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel

# 应用高斯平滑
def apply_gaussian_smoothing(gt_mask, sigma=2.0):
    kernel_size = int(2 * (sigma * 3) + 1)
    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.expand(1, 1, -1, -1)  # 扩展为适合卷积的格式

    # 使用适当的 padding 确保输出与输入形状相同
    padding = kernel_size // 2

    # 将高斯核应用于gt_mask的卷积操作
    # smoothed_mask = F.conv2d(gt_mask.unsqueeze(0), kernel, padding=padding)
    smoothed_mask = F.conv2d(gt_mask, kernel, padding=padding)
    smoothed_mask = gt_mask * 0.5 + smoothed_mask   # ****** 0.5这个比例可以调整 ******
    smoothed_mask = smoothed_mask / torch.max(smoothed_mask)
    return smoothed_mask    # 保持输出的维度与输入一致

def generate_gt_mask_batch(image_size: tuple, gt_bboxes: list, scale_factor: float = 1/8) -> torch.Tensor:
    """
    根据给定的旋转框 gt_bboxes 标注生成与缩放后的图像相同大小的 GT 掩码。

    Args:
        image_size (tuple): 原始图像的尺寸 (H, W)。
        gt_bboxes (list of torch.Tensor): 每个元素包含一个图像的旋转框标注，形状为 (N, 5),
                                        每行包含 (x, y, w, h, theta)。
        scale_factor (float): 缩放因子，用于将原始图像和 gt_bboxes 缩小。

    Returns:
        torch.Tensor: 生成的 GT 掩码，形状为 (B, H_scaled, W_scaled)，与缩放后的输入图像大小相同。
    """
    _, _, height, width = image_size
    scaled_height, scaled_width = int(height * scale_factor), int(width * scale_factor)
    batch_size = len(gt_bboxes)

    # 创建一个空白掩码图像 batch, 使用 NumPy
    masks = np.zeros((batch_size, scaled_height, scaled_width), dtype=np.uint8)

    # 遍历每个图像和相应的 gt_bboxes
    for i in range(batch_size):
        gt_bbox = gt_bboxes[i].cpu().numpy()  # 获取当前图像的 gt_bboxes

        # 缩放每个 gt_bbox
        gt_bbox[:, :4] *= scale_factor  # 缩放 (x, y, w, h)

        for bbox in gt_bbox:
            x, y, w, h, theta = bbox
            # 适当扩大gt_mask的大小以覆盖部分背景
            w = w     # [origin]: w = w + 2
            h = h   # [origin]: h = h + 2
            rect = ((x, y), (w, h), np.degrees(theta))  # OpenCV 需要角度为度数
            box = cv2.boxPoints(rect)  # 获取旋转矩形的四个顶点
            box = np.int0(box)  # 转换为整数

            # 在掩码图像上绘制多边形
            cv2.fillPoly(masks[i], [box], 1)

    # 将 NumPy 掩码转换回 PyTorch Tensor
    masks = torch.from_numpy(masks).float()
    masks = apply_gaussian_smoothing(masks.unsqueeze(1), sigma=1.0)  # 对mask进行高斯平滑

    return masks


@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # ******** 以下为自定义模块的定义 **********
        self.CFB = CascadeFusionBlock(256)
        self.SSAF = nn.ModuleList()
        for i in range(3):
            self.SSAF.append(SpectralSpatialAdaptiveFusion(256))
        self.FMG = ForegroundMaskGenerator()
        self.ActLoss = ActivationLoss()
        self.extra_conv = ConvModule(
            in_channels=256,       # 输入通道数
            out_channels=256,     # 输出通道数
            kernel_size=3,        # 卷积核大小为1x1
            stride=2,             # 步幅为1
            padding=1,            # 无需填充
            norm_cfg=dict(type='BN'),  # 使用BatchNorm进行归一化
            act_cfg=dict(type='ReLU'), # 使用ReLU激活函数
        )
        self.fem = nn.ModuleList()
        for i in range(3):
            self.fem.append(FeatureEnhancementModule(256))
        self.conv_smooth = ConvModule(
            in_channels=256,       # 输入通道数
            out_channels=256,     # 输出通道数
            kernel_size=1,        # 卷积核大小为1x1
            stride=1,             # 步幅为1
            padding=0,            # 无需填充
            norm_cfg=dict(type='BN'),  # 使用BatchNorm进行归一化
            act_cfg=dict(type='ReLU'), # 使用ReLU激活函数
        )

    @auto_fp16()
    def forward_neck(self, inputs):
        """ 将neck的forward函数写在这里方便修改 """
        assert len(inputs) == len(self.neck.in_channels)

        # build laterals,得到通道数统一为256的中间特征
        laterals = [
            lateral_conv(inputs[i + self.neck.start_level])
            for i, lateral_conv in enumerate(self.neck.lateral_convs)
        ]
        # 使用CascadeFusionBlock对顶层特征进行语义增强
        laterals[-1] = self.CFB(laterals[-1], laterals[-1])
        # build top-down path自顶向下的语义信息传递路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.SSAF[i-1](laterals[i], laterals[i - 1])   # i和i-1(更大分辨率)特征的融合
        # 上面的for循环完成 top-down 过程
        return laterals

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            # x = self.neck(x)
            x = self.forward_neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 根据原图像和gt_bboxes生成gt_mask
        # gt_masks = generate_gt_mask_batch(img.shape, gt_bboxes, scale_factor=1/4)
        # gt_masks = gt_masks.to(img.device)

        x = self.extract_feat(img)

        # 在bottom-up flow 更新特征
        # 对x[0](最低级特征计算act_loss)
        # 首先根据x[0]计算前景 mask(fore_mask)
        # fore_mask = self.FMG(x[0])
        # x[0] = self.conv_smooth(x[0] * fore_mask + x[0])  # 利用fore_mask更新x[0]
        # act_loss = self.ActLoss(fore_mask, gt_masks)  # 利用fore_mask 计算act_loss

        # used_backbone_feat = len(x)
        # 将经过act_loss refine的底层特征向上传播
        # for i in range(used_backbone_feat - 1):
        #     x[i + 1] = self.fem[i](x[i], x[i + 1])
        # # 最后利用最高层特征生成更高的两层特征
        x.append(self.extra_conv(x[-1]))
        # x.append(self.extra_conv[1](x[-1]))
        x = tuple(x)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # 把act_loss添加到losses中
        # losses['act_loss'] = act_loss
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        # 在bottom-up flow 更新特征
        # 对x[0](最低级特征计算act_loss)
        # 首先根据x[0]计算前景 mask(fore_mask)
        # fore_mask = self.FMG(x[0])
        # x[0] = self.conv_smooth(x[0] * fore_mask + x[0])  # 利用fore_mask更新x[0]

        # used_backbone_feat = len(x)
        # 将经过act_loss refine的底层特征向上传播
        # for i in range(used_backbone_feat - 1):
        #     x[i + 1] = self.fem[i](x[i], x[i + 1])
        # 最后利用最高层特征生成更高的两层特征
        x.append(self.extra_conv(x[-1]))
        # x.append(self.extra_conv[1](x[-1]))
        x = tuple(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
