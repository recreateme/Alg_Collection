import cv2
import numpy as np


def manual(img):
    # 将输入的彩色图像转换为灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    height, width = img_gray.shape  # 获取图像的高和宽
    saliencyMap = np.zeros((height, width), np.float32) # 初始化一个与灰度图像同样大小的显著性图，所有像素值初始为0

    # 遍历图像中除了边缘的每个像素（边缘像素没有四个邻居，所以从1开始，到height-1和width-1结束）
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center = img_gray[i, j]     # 获取当前像素的灰度值
            left = img_gray[i, j - 1]   # 获取当前像素左边的像素灰度值
            right = img_gray[i, j + 1]  # 获取当前像素右边的像素灰度值
            up = img_gray[i - 1, j]     # 获取当前像素上方的像素灰度值
            down = img_gray[i + 1, j]   # 获取当前像素下方的像素灰度值
            # 计算谱残差，即当前像素与其四个邻居像素的平均值之差的绝对值
            saliencyMap[i, j] = abs(center - (left + right + up + down) / 4)

    # 将显著性图的数据类型转换为uint8，并乘以255以将值范围从[0, 1]转换为[0, 255]
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # 显示显著性图并返回
    cv2.imshow("Saliency Map", saliencyMap)
    cv2.waitKey(0)
    return saliencyMap

def manual_spectral_residual_saliency(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度化

    f = np.fft.fft2(img_gray)   # 傅里叶变换
    fshift = np.fft.fftshift(f)  # 频谱中心化

    magnitude = np.log(np.abs(fshift) + 1)  # 取对数并加上1避免log(0)，得到图像频谱的幅度谱的对数
    mean_magnitude = cv2.blur(magnitude, (7, 7))  # 使用5x5的高斯模糊计算平均值
    residual = magnitude - mean_magnitude   # 计算谱残差
    fshift_r = np.exp(residual) * np.exp(1j * np.angle(fshift))  # 重构复数

    # 4. 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift_r)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 5. 显著性映射：这里简化处理，直接使用逆变换后的图像
    # 通常会包括一些后续处理，比如阈值化、标准化等
    saliencyMap = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    saliencyMap = (saliencyMap * 255).astype(np.uint8)

    # 显示显著性图
    cv2.imshow("Manual Saliency Map", saliencyMap)
    cv2.waitKey(0)
    return saliencyMap
def library(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度化
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()     # 创建OpenCV的谱残差显著性检测算法对象
    # 计算灰度图像的显著性图
    # success标志表示计算是否成功，saliencyMap是计算得到的显著性图
    (success, saliencyMap) = saliency.computeSaliency(img_gray)

    # 检查显著性图是否成功计算
    if not success:
        raise ValueError("显著性图计算失败")
    saliencyMap = (saliencyMap * 255).astype("uint8")   # 将显著性图的值从[0, 1]范围缩放到[0, 255]范围，并转换为uint8类型
    cv2.namedWindow("Saliency Map", cv2.WINDOW_AUTOSIZE)  # 创建一个窗口，并自动调整窗口大小
    cv2.imshow("Saliency Map", saliencyMap)
    cv2.waitKey(0)

    return saliencyMap



def main(img_path):
    img = cv2.imread(img_path)
    library(img)
    # manual(img)
    manual_spectral_residual_saliency(img)

if __name__ == '__main__':
    # 基于自然图像的统计原理，通过分析图像的对数谱来获得谱残差（Spectral Residual），然后进行空间变换以生成显著性图
    img_path = '../data/star.png'
    main(img_path)

