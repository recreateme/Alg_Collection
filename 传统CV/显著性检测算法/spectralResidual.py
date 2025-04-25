import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = '../data/watch.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
if image is None:
    raise ValueError("无法加载图像，请检查路径！")

# 初始化显著性检测算法
saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency_fine_grained = cv2.saliency.StaticSaliencyFineGrained_create()

# 计算显著性图
# 1. 光谱残差法
(success_spectral, saliency_map_spectral) = saliency_spectral.computeSaliency(image)
saliency_map_spectral = (saliency_map_spectral * 255).astype(np.uint8)  # 转换为 0-255

# 2. 细粒度显著性
(success_fine, saliency_map_fine) = saliency_fine_grained.computeSaliency(image)
saliency_map_fine = (saliency_map_fine * 255).astype(np.uint8)

# 可视化结果
titles = ['Original Image', 'Spectral Residual', 'Fine Grained']
images = [image, saliency_map_spectral, saliency_map_fine]

plt.figure(figsize=(12, 4))  # 调整画布大小以适应 1x3 布局
for i in range(3):
    plt.subplot(1, 3, i + 1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# 保存结果（可选）
cv2.imwrite('spectral_residual.jpg', saliency_map_spectral)
cv2.imwrite('fine_grained.jpg', saliency_map_fine)

# 对两种显著性检查结果加权并显示
weighted_saliency_map = 0.5 * saliency_map_spectral + 0.5 * saliency_map_fine
plt.imshow(weighted_saliency_map, cmap='gray')
plt.title('Weighted Saliency Map')
plt.axis('off')
plt.show()