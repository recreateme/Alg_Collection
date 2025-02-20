import cv2
import numpy as np


def band_pass_filter(image, d_low=50, d_high=100):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建带通滤波器
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= d_high ** 2
    mask_area_inside = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= d_low ** 2
    mask[mask_area & mask_area_inside] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back