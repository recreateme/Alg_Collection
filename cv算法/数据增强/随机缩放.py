import cv2

def random_scaling(image, scale_range=(0.8, 1.2)):
    """
    对输入图像进行随机缩放。
    :param image: 输入图像
    :param scale_range: 缩放比例范围，默认为0.8到1.2倍之间
    :return: 缩放后的图像
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    scaled_image = cv2.resize(image, new_size)
    # 如果需要保持原图大小，可以再调整尺寸回原来的大小
    scaled_image = cv2.resize(scaled_image, (width, height))
    return scaled_image