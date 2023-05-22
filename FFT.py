import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def evaluateSSIM(img,img_h):
    sorce, diff = compare_ssim(img,img_h,full=True)
    print(f'SSIM score : {sorce:0.2f}')

path_1 = "C:/Users/WWWONY/Desktop/pic/cameraman.tif"
img_1 = cv2.imread(path_1,cv2.IMREAD_GRAYSCALE)
f_1 = np.fft.fft2(img_1)
f_shift_1 = np.fft.fftshift(f_1)
magnitude_spectrum_1 = 20*np.log(np.abs(f_shift_1))
phase_spectrum_1 = np.angle(f_shift_1)

path_2 = "C:/Users/WWWONY/Desktop/pic/blocks.tif"
img_2 = cv2.imread(path_2,cv2.IMREAD_GRAYSCALE)
f_2 = np.fft.fft2(img_2)
f_shift_2 = np.fft.fftshift(f_2)
magnitude_spectrum_2 = 20*np.log(np.abs(f_shift_2))
phase_spectrum_2 = np.angle(f_shift_2)

#conbined = np.multiply(np.abs(f_1), np.exp(1j*np.angle(f_2)))
conbined = np.abs(f_1)
imgCombined = np.real(np.fft.ifft2(conbined))

evaluateSSIM(img_1,img_2)

plt.subplot(111)
plt.imshow(imgCombined, cmap='gray')
#plt.imsave('img A from Mag_A.jpg',imgCombined, cmap='gray')
plt.axis('off')
plt.show()
