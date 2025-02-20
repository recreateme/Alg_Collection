import cv2

# 读取左图和右图
left_img = cv2.imread("../data/left.jpg", 1)
right_img = cv2.imread("../data/right.jpg", 1)

# 检测 SIFT 特征点
sift = cv2.xfeatures2d.StarDetector()
kp1, des1 = sift.detectAndCompute(left_img, None)
kp2, des2 = sift.detectAndCompute(right_img, None)


cv2.xfeatures2d.SURF()
# 使用 BFMatcher 进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 按照距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
matched_img = cv2.drawMatches(left_img, kp1, right_img, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 保存融合后的图像
cv2.imwrite("combined.jpg", matched_img)

# 显示结果
cv2.imshow("Matched Features", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
