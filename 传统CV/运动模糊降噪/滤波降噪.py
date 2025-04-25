import cv2


gray = cv2.imread('pred.jpg', cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# 高斯滤波降噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('gaussian blur', blur)
cv2.waitKey(0)

# 均值滤波降噪
blur = cv2.blur(gray, (5, 5))
cv2.imshow('mean blur', blur)
cv2.waitKey(0)

# 双边滤波降噪
blur = cv2.bilateralFilter(gray, 9, 75, 75)
cv2.imshow('bilateral filter', blur)
cv2.waitKey(0)

# 中值滤波降噪
blur = cv2.medianBlur(gray, 5)
cv2.imshow('median blur', blur)
cv2.waitKey(0)

# 高斯双边滤波降噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
cv2.imshow('gaussian and bilateral filter', blur)
cv2.waitKey(0)

cv2.destroyAllWindows()