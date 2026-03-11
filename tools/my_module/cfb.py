import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChannelAttention", "SpatialAttention", "CascadeFusionBlock"]


class ChannelAttention(nn.Module):
    """
    Channel Attention module used in CBAM.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Channel reduction ratio.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        hidden_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, C, H, W)

        Returns:
            Tensor: channel-refined feature map
        """
        weight = self.sigmoid(self.mlp(self.avg_pool(x)))
        return x * weight


class SpatialAttention(nn.Module):
    """
    Spatial Attention module used in CBAM.

    Args:
        kernel_size (int): Convolution kernel size (3 or 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, C, H, W)

        Returns:
            Tensor: spatial-refined feature map
        """
        avg_map = torch.mean(x, dim=1, keepdim=True)
        weight = self.sigmoid(self.conv(avg_map))

        return x * weight


class CascadeFusionBlock(nn.Module):
    """
    Cascade cross-feature fusion block.

    This module performs bidirectional interaction between two feature maps
    using spatial correlation matrices.

    Args:
        channels (int): Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _normalize(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """L2 normalize tensor."""
        norm = torch.norm(x, p=2)
        return x / (norm + eps)

    def forward(self, f_h: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_h (Tensor): shape (B, C, H, W)
            f_s (Tensor): shape (B, C, H, W)

        Returns:
            Tensor: fused feature map
        """

        # Attention refinement
        f_h = f_h + self.channel_att(f_h)
        f_s = f_s + self.spatial_att(f_s)

        B, C, H, W = f_h.shape

        # reshape -> (B, C, HW)
        f_h_flat = f_h.view(B, C, -1)
        f_s_flat = f_s.view(B, C, -1)

        # Cross correlation
        M1 = torch.matmul(f_h_flat.transpose(1, 2), f_s_flat)  # (B, HW, HW)
        M1 = torch.tanh(self._normalize(M1))

        f_s_refine = torch.matmul(f_h_flat, M1) + f_s_flat

        M2 = torch.matmul(f_s_refine.transpose(1, 2), f_h_flat)
        M2 = torch.tanh(self._normalize(M2))

        f_h_refine = torch.matmul(f_s_refine, M2) + f_h_flat

        # reshape back
        f_h_refine = f_h_refine.view(B, C, H, W)
        f_s_refine = f_s_refine.view(B, C, H, W)

        # fusion
        out = torch.cat([f_h_refine, f_s_refine], dim=1)

        return self.fuse_conv(out)