import cv2
import numpy as np


def high_pass_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
    dft_shift[crow-30:crow+30, ccol-30:ccol+30] = 0 # 将中心区域置零以实现高通滤波
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

if __name__ == '__main__':
    file = r""