import cv2
import numpy as np


def manual_saliency(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)      # 高斯模糊

    # 计算图像的梯度
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化到 0 到 255 的范围
    saliency_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    saliency_map = saliency_map.astype("uint8")
    cv2.imshow("manual", saliency_map)
    cv2.waitKey(0)

    return saliency_map

def library(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imshow("library", saliencyMap)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return saliencyMap


def main(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    libary = library(img)
    manual = manual_saliency(img)
    cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    # 该方法利用空间尺度差异（center - surround  differences）来计算显著性，并使用积分图像（integralimages）实时生成高分辨率的显著性图。
    file = '../data/star.png'
    main(file)


