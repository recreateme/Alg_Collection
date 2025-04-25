import cv2
import numpy as np
from cv2 import xfeatures2d
import matplotlib.pyplot as plt


class ImageStitcher:
    def __init__(self, feature_detector='surf'):
        """
        初始化图像拼接器
        feature_detector: 'surf' 或 'sift'
        """
        self.feature_detector = feature_detector
        if feature_detector == 'surf':
            self.detector = xfeatures2d.SURF_create(
                hessianThreshold=400,
                nOctaves=4,
                nOctaveLayers=3,
                extended=True,
                upright=False
            )
        elif feature_detector == 'sift':
            self.detector = cv2.SIFT_create()

        # 创建FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect_and_compute(self, img):
        """检测特征点并计算描述符"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio=0.7):
        """匹配特征点并应用比率测试"""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def find_homography(self, kp1, kp2, matches):
        """计算单应性矩阵"""
        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask

    def blend_images(self, img1, img2, H):
        """融合两张图像"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 计算变换后图像的范围
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, H)

        pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel())

        # 平移量
        t = [-xmin, -ymin]

        # 创建平移矩阵
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        H_final = Ht.dot(H)

        # 对第一张图像进行变换
        result = cv2.warpPerspective(img1, H_final, (xmax - xmin, ymax - ymin))

        # 创建遮罩
        mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
        mask[t[1]:h2 + t[1], t[0]:w2 + t[0]] = 1

        # 平滑过渡区域
        kernel = np.ones((50, 50), np.float32) / 2500
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.dstack([mask] * 3)

        # 将第二张图像复制到结果中
        result[t[1]:h2 + t[1], t[0]:w2 + t[0]] = (
                (1 - mask[t[1]:h2 + t[1], t[0]:w2 + t[0]]) * result[t[1]:h2 + t[1], t[0]:w2 + t[0]] +
                mask[t[1]:h2 + t[1], t[0]:w2 + t[0]] * img2
        )

        return result

    def stitch(self, img1_path, img2_path):
        """执行完整的图像拼接过程"""
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError("无法读取图像")

        # 检测特征点和计算描述符
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)

        # 匹配特征点
        matches = self.match_features(desc1, desc2)

        # 计算单应性矩阵
        H, mask = self.find_homography(kp1, kp2, matches)

        if H is None:
            raise ValueError("无法找到足够的匹配点")

        # 融合图像
        result = self.blend_images(img1, img2, H)

        # 绘制匹配结果
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return result, match_img


def main():
    # 创建拼接器实例
    stitcher = ImageStitcher(feature_detector='surf')

    try:
        # 执行图像拼接
        img1_path = 'image1.jpg'  # 替换为你的第一张图像路径
        img2_path = 'image2.jpg'  # 替换为你的第二张图像路径

        result, match_img = stitcher.stitch(img1_path, img2_path)

        # 显示结果
        plt.figure(figsize=(20, 10))

        plt.subplot(211)
        plt.title('特征点匹配结果')
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(212)
        plt.title('拼接结果')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 保存结果
        cv2.imwrite('stitched_result.jpg', result)
        cv2.imwrite('matching_result.jpg', match_img)

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()