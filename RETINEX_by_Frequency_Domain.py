import numpy as np
import cv2
from matplotlib import pyplot as plt

def getImage(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return image

def makeFilter(size,sigma):
    filter = cv2.getGaussianKernel(size, sigma)
    filter2D = filter*filter.T
    return filter2D

def stretching(pix,mean,sigma):
    minValue = mean - (2*sigma)
    maxValue = mean + (2*sigma)
    if (0 <= pix and pix <= minValue):
        return 0
    elif(minValue < pix and pix <= maxValue):
        return ((pix-minValue)*255/(maxValue-minValue))
    else:
        return 255

path = "C:/Users/WWWONY/Desktop/pic/15.jpg"
img = getImage(path)
x = np.fft.fft2(img)
x = np.fft.fftshift(x)
magnitude_spectrum_1 = np.log(np.abs(x))

fil = makeFilter(640,64)
fil_fft = np.fft.fft2(fil)
fil_fft = np.fft.fftshift(fil_fft)
magnitude_spectrum_2 = np.log(np.abs(fil_fft[80:560,:])+1)
magnitude_spectrum_3 = magnitude_spectrum_1 * magnitude_spectrum_2
t = np.exp(magnitude_spectrum_3)

y = np.fft.ifftshift(t)
y = np.fft.ifft2(y)
z = np.abs(np.divide(img,y))
z_mean = np.mean(z)
z_std =np.std(z)
stVector = np.vectorize(stretching)
zout = (stVector(z, z_mean, z_std)).astype('uint8')

plt.subplot(121)
plt.imshow(magnitude_spectrum_2, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


#Homomorphic Filtering(RETINEX와 코드 유사)

# path = "C:/Users/WWWONY/Desktop/pic/15.jpg"
# img = getImage(path)
#
# x = np.fft.fft2(img)
# x = np.fft.fftshift(x)
# magnitude_spectrum_1 = np.log(np.abs(x))
#
# low_fil = makeFilter(640,64)
# high_fil = 1 - low_fil
# low_fil_fft = np.fft.fft2(low_fil)
# low_fil_fft = np.fft.fftshift(low_fil_fft)
# high_fil_fft = np.fft.fft2(high_fil)
# high_fil_fft = np.fft.fftshift(high_fil_fft)
# gamma_low = 0.5
# gamma_high = 2.0
# final_fil_fft = (gamma_low*low_fil_fft) + (gamma_high*high_fil_fft)
# magnitude_spectrum_2 = np.log(np.abs(final_fil_fft[80:560,:])+1)
# magnitude_spectrum_3 = magnitude_spectrum_1 * magnitude_spectrum_2
# t = np.exp(magnitude_spectrum_3)
#
# y = np.fft.ifftshift(t)
# y = np.fft.ifft2(y)
# z = np.abs(np.divide(img,y))
# z_mean = np.mean(z)
# z_std =np.std(z)
# stVector = np.vectorize(stretching)
# zout = (stVector(z, z_mean, z_std)).astype('uint8')
#
# plt.subplot(121)
# plt.imshow(zout, cmap='gray')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

print(f'표준편차 : {np.std(zout):0.2f}')
plt.hist(img.flatten(),256,[0,256])
plt.xlim([0,256])
plt.show()
