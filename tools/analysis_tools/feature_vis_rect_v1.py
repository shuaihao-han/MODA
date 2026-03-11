import cv2
import numpy as np
import torch
import os
import random

def featuremap_2_heatmap(feature_map, gamma=1.2, channel_sample_ratio=1.0):
    """
    将特征图转换为heatmap，可以随机选择部分通道用于可视化

    Args:
        feature_map (torch.Tensor): shape [1, C, H, W]
        gamma (float): gamma校正参数
        channel_sample_ratio (float): 取用通道比例（0~1），默认1.0表示全部通道

    Returns:
        np.ndarray: heatmap [H, W]
    """
    assert isinstance(feature_map, torch.Tensor)
    assert feature_map.ndim == 4, "feature_map must be in shape [1, C, H, W]"
    feature_map = feature_map.detach().cpu()

    # 选择部分通道
    B, C, H, W = feature_map.shape
    if channel_sample_ratio < 1.0:
        num_channels = int(C * channel_sample_ratio)
        sampled_indices = sorted(random.sample(range(C), num_channels))
        feature_map = feature_map[:, sampled_indices, :, :]

    # 按通道求和并对 batch 取平均
    heatmap = torch.sum(feature_map, dim=1).numpy()  # shape: [1, H, W]
    heatmap = np.mean(heatmap, axis=0)  # shape: [H, W]

    # Gamma 校正
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = np.power(heatmap, gamma)

    return heatmap


def draw_feature_map(features, save_dir='/data1/users/hanshuaihao01/mmrotate/work_dir/feat_0804/00871', img=None, name="feat1", channel_sample_ratio=1.0):
    """
    可视化特征图并保存，支持通道随机采样

    Args:
        features (Tensor): 特征图，shape [B, C, H, W]
        save_dir (str): 保存路径
        img (Tensor): 原图像，shape [B, 5, H, W]，可为 None
        name (str): 保存文件名
        channel_sample_ratio (float): 取用通道比例（0~1），默认1.0表示全部通道
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(features, torch.Tensor):
        for i, heat_map_tensor in enumerate(features):
            if img is not None:
                img_tensor = img[i]
            else:
                img_tensor = None

            # 加上 batch 维度 [C, H, W] -> [1, C, H, W]
            heat_map_tensor = heat_map_tensor.unsqueeze(0)

            # 特征图转 heatmap（支持通道采样）
            heatmap = featuremap_2_heatmap(heat_map_tensor, gamma=1.1, channel_sample_ratio=channel_sample_ratio)

            # Resize heatmap
            heatmap = cv2.resize(heatmap, (1216, 928))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 原图处理
            if img_tensor is not None:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                if img_np.shape[2] >= 5:
                    img_np = img_np[:, :, [1, 2, 4]]  # 使用第2,3,5通道
                else:
                    img_np = img_np[:, :, :3]  # 使用前3通道

                img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8) * 255
                img_np = cv2.resize(img_np.astype(np.uint8), (1216, 928))

                # 叠加可视化
                superimposed_img = cv2.addWeighted(heatmap, 0.4, img_np, 0.5, 0)
            else:
                superimposed_img = heatmap

            # 保存
            save_path = os.path.join(save_dir, name)
            cv2.imwrite(save_path, superimposed_img)
