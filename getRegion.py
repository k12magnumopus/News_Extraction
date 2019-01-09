import cv2
import numpy as np
from matplotlib import pyplot as plt
import  imutils
import random


img = cv2.imread('Test_Image.jpeg')
img = imutils.resize(img, height=500)
# x, y, height, width = cv2.boundingRect(contour[i])

canvas = img.copy()
img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(5,5),0)
cannyImg = cv2.Canny(img2gray,100,200)


kernel = np.ones((5,5), np.uint8)

# img_dil = cv2.dilate(cannyImg, kernel, iterations=1)
img_dil2 = cv2.dilate(cannyImg, kernel, iterations=2)


contours, hierarchy = cv2.findContours(img_dil2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours
# imgCon = cv2.drawContours(img_dil2, [cnt], 0, (0,255,0), 3)

# imgCon = cv2.drawContours(img_dil2, contours, -1, (0,255,0), 3)

for c in cnt:
    perimeter = cv2.arcLength(c, True)
    # epsilon = 0.01* cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.07*perimeter, True)
    #
    # cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 3)
    if len(approx) ==4:
        rect = cv2.boundingRect(c)
        x, y, w, h = [r for r in rect]
        b, g = random.sample(range(0, 255), 2)
        cv2.rectangle(canvas, (x, y), ((x + w), (y + h)), (b, g, 255), 3)
        # cv2.drawContours(canvas, [approx], 0, (0, 0, 255), 2)

# cv2.imshow('dilation', img_dil)
cv2.imshow('dilation2', canvas)
# cv2.imshow('dst_rt', cannyImg)
cv2.waitKey(0)
cv2.destroyAllWindows()