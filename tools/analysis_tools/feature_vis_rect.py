import cv2
import numpy as np
import torch
import os

def featuremap_2_heatmap(feature_map, gamma=1.2):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = torch.sum(feature_map, dim=1).cpu().numpy()  # Summing along channels
    heatmap = np.mean(heatmap, axis=0)  # Average across batch if needed

    # Apply gamma correction for better contrast
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.power(heatmap, gamma)  # Gamma adjustment

    return heatmap


def draw_feature_map(features, save_dir='/data3/hanshuaihao01/work_dirs_backup/171/mmrotate/work_dirs/feature_map/best/feat/', img=None, name="feat1"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(features, torch.Tensor):
        for i, heat_map_tensor in enumerate(features):
            if img is not None:
                img_tensor = img[i]
            else:
                img_tensor = None

            heat_map_tensor = heat_map_tensor.unsqueeze(0)
            heatmap = featuremap_2_heatmap(heat_map_tensor, gamma=1.1)  # Customize gamma here  [origin]
            # heatmap = featuremap_2_heatmap(heat_map_tensor, gamma=1)
            # Resize heatmap to desired size
            heatmap = cv2.resize(heatmap, (1216, 928))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  #  [added]
            # threshold = 0.2
            # heatmap[heatmap < threshold] = 0
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Try alternative color map

            # Prepare the original image if provided
            if img_tensor is not None:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()[:, :, [1, 2, 4]]  # Select desired channels
                # img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, [1, 2, 4]]  # Select desired channels
                img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255
                img_np = cv2.resize(img_np.astype(np.uint8), (1216, 928))

                # Combine heatmap and image with adjustable alpha for better contrast
                superimposed_img = cv2.addWeighted(heatmap, 0.4, img_np, 0.5, 0)    # [origin]: superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
            else:
                superimposed_img = heatmap

            # Save the result
            cv2.imwrite(os.path.join(save_dir, name), superimposed_img)
