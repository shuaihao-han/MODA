import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AdaptiveSpectralFeatureRefinementEuclidean",
           "AdaptiveSpectralFeatureRefinementCosine",
           "AdaptiveSpectralFeatureRefinementCombined",
           "EdgeEnhanceSpatialFeatureRefinement",
           "SpectralSpatialAdaptiveFusion"]


class AdaptiveSpectralFeatureRefinementEuclidean(nn.Module):
    """
    ASFR module using Euclidean distance for local patch feature refinement.
    """

    def __init__(self, patch_size: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)

    def forward(self, fe_lv: torch.Tensor, fused_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fe_lv.shape

        # unfold local patches
        patches = self.unfold(fused_features)  # (B, C*patch_size*patch_size, H*W)
        patches = patches.reshape(B, C, self.patch_size * self.patch_size, H * W)  # (B, C, k*k, H*W)
        fe_lv_flat = fe_lv.reshape(B, C, H * W)

        # Euclidean distance similarity
        distances = torch.norm(
            patches.permute(0, 3, 2, 1) - fe_lv_flat.permute(0, 2, 1).unsqueeze(2),
            dim=-1
        )  # (B, H*W, k*k)

        weights = F.softmax(-distances, dim=-1)
        refined = torch.matmul(weights.unsqueeze(2),
                               patches.permute(0, 3, 2, 1).reshape(B, H * W, self.patch_size * self.patch_size, C))
        refined = refined.squeeze(2).permute(0, 2, 1).reshape(B, C, H, W)

        return refined + fe_lv  # residual


class AdaptiveSpectralFeatureRefinementCosine(nn.Module):
    """
    ASFR module using Cosine similarity for local patch feature refinement.
    """

    def __init__(self, patch_size: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)

    def forward(self, fe_lv: torch.Tensor, fused_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fe_lv.shape

        patches = self.unfold(fused_features).reshape(B, C, self.patch_size * self.patch_size, H * W)
        fe_lv_flat = fe_lv.reshape(B, C, H * W)

        patches_norm = F.normalize(patches, dim=1)
        fe_lv_norm = F.normalize(fe_lv_flat, dim=1)

        cosine_sim = torch.einsum(
            'bhkc,bhc->bhk',
            patches_norm.permute(0, 3, 2, 1),
            fe_lv_norm.permute(0, 2, 1)
        )

        weights = F.softmax(cosine_sim, dim=-1)
        refined = torch.matmul(weights.unsqueeze(2),
                               patches.permute(0, 3, 2, 1).reshape(B, H * W, self.patch_size * self.patch_size, C))
        refined = refined.squeeze(2).permute(0, 2, 1).reshape(B, C, H, W)

        return refined + fe_lv


class AdaptiveSpectralFeatureRefinementCombined(nn.Module):
    """
    Combined ASFR module using both Euclidean distance and Cosine similarity.
    """

    def __init__(self, patch_size: int = 3, fusion_method: str = 'add', alpha: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2)
        self.fusion_method = fusion_method
        self.alpha = alpha

    def forward(self, fe_lv: torch.Tensor, fused_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fe_lv.shape
        patches = self.unfold(fused_features).reshape(B, C, self.patch_size * self.patch_size, H * W)
        fe_lv_flat = fe_lv.reshape(B, C, H * W)

        # Cosine similarity
        patches_norm = F.normalize(patches, dim=1)
        fe_lv_norm = F.normalize(fe_lv_flat, dim=1)
        cosine_sim = torch.einsum('bhkc,bhc->bhk',
                                  patches_norm.permute(0, 3, 2, 1),
                                  fe_lv_norm.permute(0, 2, 1))
        cosine_weights = F.softmax(cosine_sim, dim=-1)

        # Euclidean similarity
        distances = torch.norm(patches.permute(0, 3, 2, 1) - fe_lv_flat.permute(0, 2, 1).unsqueeze(2), dim=-1)
        euclidean_weights = F.softmax(-distances, dim=-1)

        # Fusion
        if self.fusion_method == 'add':
            weights = self.alpha * cosine_weights + (1 - self.alpha) * euclidean_weights
        else:  # concat
            weights = F.softmax(torch.cat([cosine_weights, euclidean_weights], dim=-1), dim=-1)

        patches_reshaped = patches.permute(0, 3, 2, 1).reshape(B, H * W, self.patch_size * self.patch_size, C)
        if self.fusion_method == 'add':
            refined = torch.matmul(weights.unsqueeze(2), patches_reshaped)
        else:  # concat
            refined_cos = torch.matmul(cosine_weights.unsqueeze(2), patches_reshaped)
            refined_euc = torch.matmul(euclidean_weights.unsqueeze(2), patches_reshaped)
            refined = torch.cat([refined_cos, refined_euc], dim=-1)

        refined = refined.squeeze(2).permute(0, 2, 1).reshape(B, -1, H, W)
        return refined + fe_lv


class StripConvBlock(nn.Module):
    """Horizontal + vertical strip convolutions for feature refinement."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.h_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size),
                                padding=(0, kernel_size // 2), bias=False)
        self.v_conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                                padding=(kernel_size // 2, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h_conv(x) + self.v_conv(x)


class EdgeEnhanceSpatialFeatureRefinement(nn.Module):
    """Edge-enhancing spatial feature refinement (ESFR) module."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-frequency
        F_L = self.avgpool(x)
        F_L = F.interpolate(self.conv3(F_L), size=x.shape[2:], mode='bilinear', align_corners=True)

        # High-frequency
        F_H = x - F.interpolate(self.avgpool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        weight = torch.sigmoid(self.conv1_1(F_H))
        F_H = F_H * weight + F_H

        # Fuse
        out = self.conv1_2(torch.cat([F_H, F_L], dim=1))
        return out + x


class SpectralSpatialAdaptiveFusion(nn.Module):
    """
    Spectral-Spatial Adaptive Fusion (SSAF) module.
    Fuses high-spectral and spectral-angle features adaptively.
    """

    def __init__(self, in_channels: int, patch_size: int = 3):
        super().__init__()
        self.asfr = AdaptiveSpectralFeatureRefinementEuclidean(patch_size=patch_size)
        self.esfr = EdgeEnhanceSpatialFeatureRefinement(in_channels)
        self.adp_conv = nn.Conv2d(2, 2, 1, bias=False)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fh: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        # ASFR on hyperspectral feature
        fh = self.asfr(fh, fh)
        fh = F.interpolate(fh, size=fs.shape[2:], mode='bilinear', align_corners=True)

        # ESFR on spectral angle feature
        fs = self.esfr(fs)

        # Adaptive fusion
        fh_avg, fs_avg = fh.mean(1, keepdim=True), fs.mean(1, keepdim=True)
        weights = torch.sigmoid(self.adp_conv(torch.cat([fh_avg, fs_avg], dim=1)))  # (B,2,H,W)
        out = weights[:, 0:1] * fh + weights[:, 1:2] * fs

        return self.fuse_conv(out)