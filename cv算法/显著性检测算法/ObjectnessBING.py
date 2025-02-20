import cv2
import numpy as np


def compute_objectness(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    grad_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
    grad_a = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=5)
    grad_b = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=5)

    mag_l = cv2.normalize(np.sqrt(grad_l ** 2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_a = cv2.normalize(np.sqrt(grad_a ** 2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_b = cv2.normalize(np.sqrt(grad_b ** 2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined_gradient = cv2.addWeighted(mag_l, 0.33, mag_a, 0.33, 0)
    combined_gradient = cv2.addWeighted(combined_gradient, 1, mag_b, 0.33, 0)

    _, objectness_map = cv2.threshold(combined_gradient, 50, 255, cv2.THRESH_BINARY)

    cv2.imshow("Manual Objectness Map", objectness_map)
    cv2.waitKey(0)
    return objectness_map


def library_objectness(image):
    objectness_bing = cv2.saliency.ObjectnessBING_create()

    if objectness_bing is None:
        print("ObjectnessBING 对象创建失败。")
        return None
    else:
        (success, saliency_map) = objectness_bing.computeSaliency(image)
        if success:
            cv2.imshow("ObjectnessBING Saliency Map", saliency_map)
            cv2.waitKey(0)
            return saliency_map
        else:
            print("ObjectnessBING 显著性计算失败。")
            return None


def main(filename):
    img = cv2.imread(filename)
    if img is None:
        raise ValueError("无法读取图像，请检查文件路径。")

    library_objectness(img)
    # compute_objectness(img)  # 可以选择手动计算显著性图
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 基于二值化梯度特征（BING features）进行物体检测
    file = "../data/social.jpg"  # 请根据实际文件路径修改
    main(file)
