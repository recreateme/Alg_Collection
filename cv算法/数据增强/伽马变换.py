import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):
    """
    对输入图像进行伽玛校正。
    :param image: 输入图像
    :param gamma: 伽玛值，默认为1.0表示不做任何改变
    :return: 校正后的图像
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
