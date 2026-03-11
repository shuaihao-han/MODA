import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    # for c in range(feature_map.shape[1]):
    #     heatmap += feature_map[:,c,:,:]
    # 将上面的循环代码进行优化
    heatmap = torch.sum(feature_map, dim=1)
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = '/data1/users/hanshuaihao01/mmrotate/work_dirs/feature_map/oriented_reppoints_P2_img_resize',
                     img = None, name = "feat1"):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (928, 1216))
            for heatmap in heatmaps:
                # 将 heatmap 进行 resize 以匹配图像的尺寸
                heatmap = cv2.resize(heatmap, (1216, 928))
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap
                # 将 img 由 GPU上的 tensor 转到CPU上的numpy
                img = img.squeeze(0)
                # 将 img 转换到 (H, W, C), 并抽取其中的可见光通道
                img = img.permute(1, 2, 0)[:, :, [1, 2, 4]]
                img = img.cpu().numpy()
                # 将 img 归一化到0~1, 再将其 * 255
                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                # 将 img 进行 resize 以匹配 heatmap 的尺寸
                # img = cv2.resize(img, (304, 232))
                # 可以适当调节下面的比例了来实现更好的可视化效果
                superimposed_img = heatmap * 0.5 + img * 0.5
                # superimposed_img = heatmap * 0.6 + img * 0.4
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                cv2.imwrite(os.path.join(save_dir, name), superimposed_img)
                i=i+1

    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存,使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                i=i+1
