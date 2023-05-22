import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def getA(path):
    img_a = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return img_a

def getB(img_a):
    mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    lapla = cv2.filter2D(img_a, -1, mask)
    lapla = cv2.add(lapla, 127)
    return lapla

def getC(img_a):
    mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img_b = cv2.filter2D(img_a, -1, mask)
    img_c = cv2.add(img_a,img_b)                        #단순히 +연산을 진행하면 Overflow가 발생하기 때문에 함수를 사용
    return img_c

def getD(img_a):
    sobelx = cv2.Sobel(img_a, -1,1,0, ksize=3)
    sobely = cv2.Sobel(img_a, -1,0,1, ksize=3)
    sobel = cv2.add(sobelx,sobely)
    return sobel

def getE(img_d):
    kernel = np.ones((5,5),np.float32)/25
    img_e = cv2.filter2D(img_d,-1,kernel).astype('uint8')
    #img_e = cv2.normalize(img_e, None, 127, 255, cv2.NORM_MINMAX)
    return img_e

def getF(img_e,img_c):
    img_e = cv2.normalize(img_e,None,0,20,cv2.NORM_MINMAX)
    img_c = cv2.normalize(img_c,None,0,20,cv2.NORM_MINMAX)
    img_f = cv2.multiply(img_e,img_c).astype('uint8')
    return img_f

def getG(img_f,img_a):
    img_g = cv2.add(img_f,img_a)
    return img_g

def getH(img_g,r):
    img_h = (((img_g/255)**r)*255).astype('uint8')
    return img_h

def evaluateSSIM(img,img_h):
    sorce, diff = compare_ssim(img,img_h,full=True)
    print(f'SSIM score : {sorce:0.4f}')

path = "C:/Users/WWWONY/Desktop/pic/Fig1.06(b).jpg"

a_image = getA(path)
b_image = getB(a_image)
c_image = getC(a_image)
d_image = getD(a_image)
e_image = getE(d_image)
f_image = getF(e_image,c_image)
g_image = getG(f_image,a_image)
h_image = getH(g_image,0.45)

a_eq = cv2.equalizeHist(a_image)

evaluateSSIM(g_image,h_image)

cv2.imshow('image',f_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.hist(g_image.flatten(),256,[0,256])
plt.xlim([0,256])
plt.show()