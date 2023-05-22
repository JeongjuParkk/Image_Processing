import numpy as np
import cv2

def nonmax_suppression(sobel, direct):
    rows, cols = sobel.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            values = sobel[i-1:i+2, j-1:j+2].flatten()
            first = [3, 0, 1, 2]
            id = first[direct[i, j]]
            v1, v2 = values[id], values[8-id]
            dst[i, j] = sobel[i, j] if (v1 < sobel[i, j] > v2) else 0
    return dst

def trace(max_sobel, i, j, low):
    h, w = max_sobel.shape
    if (0 <= i < h and 0 <= j < w) == False: return
    if pos_ck[i, j] > 0 and max_sobel[i, j] > low:
        pos_ck[i, j] = 255
        canny[i, j] = 255

        trace(max_sobel, i-1, j-1, low)
        trace(max_sobel, i, j-1, low)
        trace(max_sobel, i+1, j-1, low)
        trace(max_sobel, i-1, j, low)
        trace(max_sobel, i+1, j, low)
        trace(max_sobel, i-1, j+1, low)
        trace(max_sobel, i, j+1, low)
        trace(max_sobel, i+1, j+1, low)

def hysteresis_th(max_sobel, low, high):
    rows, cols = max_sobel.shape[:2]
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if max_sobel[i, j] >= high: trace(max_sobel, i, j, low)

image = cv2.imread("C:/Users/WWWONY/Desktop/pic/mine1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None: raise Exception("영상파일 읽기 오류")

pos_ck = np.zeros(image.shape[:2], np.uint8)
canny = np.zeros(image.shape[:2], np.uint8)

# 캐니 에지 검출
gaus_img = cv2.GaussianBlur(image, (5, 5), 64)
cv2.imshow("gaussian.jpg", gaus_img)
Gx = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 1, 0, 3)
Gy = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 0, 1, 3)

sobel = cv2.magnitude(Gx, Gy)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

directs = cv2.phase(Gx, Gy) / (np.pi/4)
directs = directs.astype(int) % 4
max_sobel = nonmax_suppression(sobel, directs)
max_sobel = max_sobel.astype(np.uint8)

cv2.imshow("nonmax_suppression.jpg", max_sobel)

checker = sobel >= max_sobel
unique, counts = np.unique(checker, return_counts=True)
checker = dict(zip(unique, counts))

m = 0
n = 0

##################
nonmax = max_sobel.copy()

hysteresis_th(max_sobel, 50, 150)

canny = max_sobel.copy()
canny2 = cv2.Canny(image, 1500, 5000,apertureSize=5)

cv2.imshow("image", image)
cv2.imshow("sobel", sobel)
cv2.imshow("canny", canny)
cv2.imshow("OpenCV_Canny", canny2)


# 허프 변환
lines = cv2.HoughLines(canny2,1,np.pi/180,120)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)


cv2.imshow('hough', image)
cv2.waitKey(0)