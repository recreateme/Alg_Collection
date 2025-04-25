import cv2
import numpy as np
from scipy.ndimage import rotate

def random_rotation(image, angle_range=(-20, 20)):
    """
    对输入图像进行随机旋转。
    :param image: 输入图像
    :param angle_range: 旋转角度范围，默认为-20到20度之间
    :return: 旋转后的图像
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image