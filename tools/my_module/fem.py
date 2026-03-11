import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureEnhancementModule"]


class FeatureEnhancementModule(nn.Module):
    """
    Feature Enhancement Module.

    This module enhances two feature branches by modeling channel-level
    cross-interactions using global descriptors.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        # downsample low-level feature
        self.downsample = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1, bias=False
        )

        # global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # channel projection
        self.fc_low = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.fc_high = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # fusion conv
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _channel_cross_attention(
        self, low_desc: torch.Tensor, high_desc: torch.Tensor
    ):
        """
        Compute channel-wise cross-attention.

        Args:
            low_desc (Tensor): shape (B, C, 1, 1)
            high_desc (Tensor): shape (B, C, 1, 1)

        Returns:
            Tuple[Tensor, Tensor]: refined channel descriptors
        """

        B, C, _, _ = low_desc.shape

        # reshape -> (B, C, 1)
        low_vec = low_desc.reshape(B, C, 1)
        high_vec = high_desc.reshape(B, C, 1)

        # cross similarity matrix (B, C, C)
        weight = torch.matmul(low_vec, high_vec.transpose(1, 2))

        weight = F.softmax(weight, dim=-1)

        # channel refinement
        low_refined = torch.matmul(weight, low_vec).reshape(B, C, 1, 1)
        high_refined = torch.matmul(weight.transpose(1, 2), high_vec).reshape(B, C, 1, 1)

        return low_refined, high_refined

    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_low (Tensor): low-level feature (B, C, H, W)
            feat_high (Tensor): high-level feature (B, C, H/2, W/2)

        Returns:
            Tensor: enhanced feature map
        """

        # align resolution
        feat_low = self.downsample(feat_low)

        # global channel descriptors
        low_desc = self.fc_low(self.global_pool(feat_low))
        high_desc = self.fc_high(self.global_pool(feat_high))

        # cross-channel attention
        low_weight, high_weight = self._channel_cross_attention(
            low_desc, high_desc
        )

        # feature modulation
        feat_low = feat_low * low_weight
        feat_high = feat_high * high_weight

        # fusion
        out = self.fuse(feat_low + feat_high)

        return out