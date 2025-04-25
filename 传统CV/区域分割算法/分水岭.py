import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../data/star.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊

# 使用 Otsu 阈值法进行二值化
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 进行形态学开运算，去除小的噪声点
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀

# 确定前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通区域
_, markers = cv2.connectedComponents(sure_fg)

# 所有标记加 1，确保背景标记为 1
markers = markers + 1

# 将未知区域标记为 0
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # 将分水岭线标记为红色

# 显示结果
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
