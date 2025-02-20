import cv2


def reflection(image, horizontal=True):
    """
    对输入图像进行反射变换。
    :param image: 输入图像
    :param horizontal: 是否水平翻转，默认True
    :return: 翻转后的图像
    """
    if horizontal:
        flipped_image = cv2.flip(image, 1) # 水平翻转
    else:
        flipped_image = cv2.flip(image, 0) # 垂直翻转
    return flipped_image