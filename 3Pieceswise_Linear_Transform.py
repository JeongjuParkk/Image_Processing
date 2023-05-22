import cv2
import numpy as np
from matplotlib import pyplot as plt

#실제 계산을 하는 함수 생성
def get3pieceTrans(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

def getGammaTrans(pix, r1, s1, r2, s2,gamma):
    if (0 <= pix and pix <= r1):
        return s1
    elif (r1 < pix and pix <= r2):
        return (((((s2 - s1) / (r2 - r1)) * (pix - r1))/255)**gamma)*255 + s1
    else:
        return s2

#이미지 경로 지정과 이미지 데이터 불러오기
path = "C:/Users/WWWONY/Desktop/pic/4627333807.jpg"
img = cv2.imread(path)

#이미지의 밝기별 픽셀 수를 표기
unique, counts = np.unique(img, return_counts=True)
print (np.asarray((unique, counts)))

#parameter 설정
r1 = 130
s1 = 30
r2 = 180
s2 = 224
gamma = 1.2

#연산 실행
Data4Mapping = np.vectorize(get3pieceTrans)                  #Numpy의 기본 형태인 array에 mapping을 하기 위해 사용
ThrPieceTrans = Data4Mapping(img, r1, s1, r2, s2)
Data4Mapping_gam = np.vectorize(getGammaTrans)
GammaTrans = Data4Mapping_gam(img, r1, s1, r2, s2, gamma)

#이미지 프로세싱 결과물들을 비교하기 위해 한줄로 나열
Output = cv2.hconcat([img.astype('uint8'),ThrPieceTrans.astype('uint8') ,GammaTrans.astype('uint8')])

#결과값 출력
cv2.imshow('output', Output)
cv2.waitKey(0)
cv2.destroyAllWindows()

#결과값들의 히스토그램 출력
imgHist = img.flatten()
thrPieceHist = ThrPieceTrans.flatten()
gammaHist = GammaTrans.flatten()
plt.hist([imgHist,thrPieceHist,gammaHist],256,[0,256],alpha = 0.7,color=['orange','red','blue'],label=['Origina','3piecewise','gamma'])
plt.title('Compare Histogram Image')
plt.legend()
plt.xlim([0,256])
plt.show()

#히스토그램의 분산 출력
Img_var = np.var(imgHist)
Piece_var = np.var(thrPieceHist)
Gamma_var = np.var(gammaHist)
print(Img_var,Piece_var,Gamma_var)