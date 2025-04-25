import cv2
import numpy as np

def low_pass_filter(image, d=20):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-d:crow+d, ccol-d:ccol+d] = 1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back