import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageInpainter:
    def __init__(self):
        self.drawing = False    # 鼠标是否按下
        self.mask = None        # 修复掩码
        self.img = None         # 输入图像
        self.brush_size = 5      # 画笔大小


    def draw_mask(self, event, x, y, flags, param):  # 鼠标事件处理函数
        """鼠标事件处理函数，用于绘制修复掩码"""
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
            self.drawing = True  # 开始绘制
        elif event == cv2.EVENT_MOUSEMOVE:          # 鼠标移动
            if self.drawing:
                cv2.circle(self.mask, (x, y), self.brush_size, (255, 255, 255), -1)        # 绘制圆形
                cv2.circle(self.display_img, (x, y), self.brush_size, (0, 0, 255), -1)       # 绘制圆形
        elif event == cv2.EVENT_LBUTTONUP:          # 左键松开
            self.drawing = False    # 停止绘制

    def create_mask_interactive(self, image_path):
        """交互式创建修复掩码"""
        # 读取图像
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError("无法读取图像")

        # 创建掩码和显示图像
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.display_img = self.img.copy()

        # 创建窗口并设置鼠标回调
        cv2.namedWindow('Draw Mask')
        cv2.setMouseCallback('Draw Mask', self.draw_mask)

        print("使用鼠标在要修复的区域上绘制。按 'r' 重置掩码，按 'q' 开始修复")

        while True:
            cv2.imshow('Draw Mask', self.display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # 按q退出
                break
            elif key == ord('r'):  # 按r重置
                self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                self.display_img = self.img.copy()

        cv2.destroyAllWindows()
        return self.img, self.mask

    def inpaint_image(self, img, mask, method='TELEA', radius=3):
        """
        使用指定方法进行图像修复

        参数:
        img: 输入图像
        mask: 修复掩码
        method: 修复方法 ('NS' 或 'TELEA')
        radius: 修复半径
        """
        if method == 'NS':
            result = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
        else:  # TELEA
            result = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
        return result

    def create_damage_mask(self, img, damage_type='random', intensity=0.1):
        """创建模拟的损坏掩码"""
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if damage_type == 'random':
            # 随机噪点
            mask_points = np.random.random(img.shape[:2]) < intensity
            mask[mask_points] = 255
        elif damage_type == 'lines':
            # 随机线条
            for _ in range(int(20 * intensity)):
                pt1 = (np.random.randint(0, img.shape[1]),
                       np.random.randint(0, img.shape[0]))
                pt2 = (np.random.randint(0, img.shape[1]),
                       np.random.randint(0, img.shape[0]))
                cv2.line(mask, pt1, pt2, 255, 2)
        elif damage_type == 'text':
            # 模拟文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Damage Text"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            x = (img.shape[1] - textsize[0]) // 2
            y = (img.shape[0] + textsize[1]) // 2
            cv2.putText(mask, text, (x, y), font, 1, 255, 2)

        return mask


def demonstrate_inpainting():
    """演示不同类型的图像修复"""
    # 创建修复器实例
    inpainter = ImageInpainter()

    # 读取测试图像
    img_path = './data/star.png'  # 替换为你的图像路径
    img = cv2.imread(img_path)

    if img is None:
        print("无法读取图像")
        return

    # 创建不同类型的损坏掩码
    damage_types = ['random', 'lines', 'text']

    plt.figure(figsize=(15, 5))

    for i, damage_type in enumerate(damage_types):
        # 创建损坏掩码
        mask = inpainter.create_damage_mask(img, damage_type)

        # 使用两种方法进行修复
        result_ns = inpainter.inpaint_image(img, mask, method='NS')
        result_telea = inpainter.inpaint_image(img, mask, method='TELEA')

        # 显示结果
        plt.subplot(2, 3, i + 1)
        plt.title(f'Damaged ({damage_type})')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, i + 4)
        plt.title(f'Repaired (TELEA)')
        plt.imshow(cv2.cvtColor(result_telea, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def interactive_inpainting():
    """交互式图像修复演示"""
    inpainter = ImageInpainter()

    try:
        # 读取图像并交互式创建掩码
        img_path = './data/social.jpg'  # 替换为你的图像路径
        img, mask = inpainter.create_mask_interactive(img_path)

        # 使用两种方法进行修复
        result_ns = inpainter.inpaint_image(img, mask, method='NS')
        result_telea = inpainter.inpaint_image(img, mask, method='TELEA')

        # 显示结果
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(132)
        plt.title('Navier-Stokes')
        plt.imshow(cv2.cvtColor(result_ns, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(133)
        plt.title('TELEA')
        plt.imshow(cv2.cvtColor(result_telea, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    # 演示自动损坏和修复
    demonstrate_inpainting()

    # 演示交互式修复
    # interactive_inpainting()