import numpy as np
import cv2
from matplotlib import pyplot as plt

def getImage(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return image

def padding_mirror(filter_size,image):
    padding_size = int(filter_size/2)
    output = np.pad(image,(padding_size,padding_size),'reflect')
    return output

def makeFilter(size,sigma):
    filter = cv2.getGaussianKernel(size, sigma)
    #filter2D = np.outer(filter,filter.transpose())
    filter2D = filter*filter.T
    return filter2D

def applyFilter(image,filter):
    output = cv2.filter2D(image,-1,filter,borderType=cv2.BORDER_REFLECT_101)
    return output

def stretching(pix,mean,sigma):
    minValue = mean - (2*sigma)
    maxValue = mean + (2*sigma)
    if (0 <= pix and pix <= minValue):
        return 0
    elif(minValue < pix and pix <= maxValue):
        return ((pix-minValue)*255/(maxValue-minValue))
    else:
        return 255

path = "C:/Users/WWWONY/Desktop/pic/www.jpg"
img = getImage(path)
filter_size = 424
sigma = [16,32,64]
x = padding_mirror(filter_size,img)
print(f'표준편차 : {np.std(img):0.2f}')

for i in sigma:
    gaussian_filter = makeFilter(filter_size,i)
    y = applyFilter(img,gaussian_filter)
    z = img/y
    z_mean = np.mean(z)
    z_std =np.std(z)
    stVector = np.vectorize(stretching)
    zout = (stVector(z, z_mean, z_std)).astype('uint8')
    print(f'simga : {i}, 표준편차 : {np.std(zout):0.2f}')
    plt.hist(zout.flatten(), 256, [0, 256])
    plt.xlim([0, 256])
    plt.show()
    cv2.imshow(f'RETINEX_15_sigma{i}.jpg', zout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()