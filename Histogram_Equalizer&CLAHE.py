import numpy as np
from matplotlib import pyplot as plt
import cv2

def getImage(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return img

def getCDF(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    CDF = hist.cumsum()
    CDF_m = np.ma.masked_equal(CDF,0)                       #값이 0인 부분은 --으로 처리해서 마스크
    CDF_m = (CDF_m - CDF_m.min()) * 255 / (CDF_m.max() - CDF_m.min())
    CDF = np.ma.filled(CDF_m,0).astype('uint8')             #--로 마스크 처리된 부분을 다시 0으로 복원
    return CDF

def evaluateImage(img_hist):
    value = np.std(img_hist)
    return value

def CLAHE(img, Criteria):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    value = 0
    for i in range(0,256):
        if hist[i] > Criteria:
            value = value + (hist[i] - Criteria)
            hist[i] = Criteria
    for j in range(0,256):
        hist[j] += value//255
    CDF = hist.cumsum()
    CDF_m = np.ma.masked_equal(CDF, 0)                          # 값이 0인 부분은 --으로 처리해서 마스크
    CDF_m = (CDF_m - CDF_m.min()) * 255 / (CDF_m.max() - CDF_m.min())
    CDF = np.ma.filled(CDF_m, 0).astype('uint8')                # --로 마스크 처리된 부분을 다시 0으로 복원
    return CDF


a = getImage("C:/Users/WWWONY/Desktop/pic/13.jpg")
a_CDF = getCDF(a)
img1 = a_CDF[a]
CL_CDF = CLAHE(a,3000)
img2 = CL_CDF[a]


before_evaluation = evaluateImage(a)
after_evaluation = evaluateImage(img1)
after_CLAHE = evaluateImage(img2)


print(f'원본 표준편차 : {before_evaluation:0.2f}\n'
      f'평활화 이후 표준편차 : {after_evaluation:0.2f}\n'
      f'CLAHE 이후 표준편차 : {after_CLAHE:0.2f}')

cv2.imshow('img', a)
plt.hist(a.flatten(),256,[0,256])
plt.xlim([0,256])
plt.show()

cv2.imshow('img', img1)
plt.hist(img1.flatten(),256,[0,256])
plt.xlim([0,256])
plt.show()

cv2.imshow('img', img2)
plt.hist(img2.flatten(),256,[0,256])
plt.xlim([0,256])
plt.show()

'''
기준값 변화하면서 이미지와 데이터를 자동으로 저장
for i in range(2,13,2):
    CL_CDF = CLAHE(a,i*1000)
    img2 = CL_CDF[a]
    after_CLAHE = evaluateImage(img2)
    print(f'{i*1000}을 기준으로 설정했을 때 표준편차 : {after_CLAHE:0.2f}')
    cv2.imshow('img', img2)
    cv2.imwrite(f'Criteria{i*1000}.jpg',img2)
    plt.hist(img2.flatten(), 256, [0, 256])
    plt.title(f'Criteria{i*1000}')
    plt.xlim([0, 256])
    plt.savefig(f'Criteria{i*1000}.png')
    plt.cla()
'''
