#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:41:29 2020

@author: lizeth
"""

import cv2 as cv
import numpy as np



img=cv.imread('t3.jpg',0)


rows,cols = img.shape
imgR = np.zeros((rows,cols), dtype = "uint8") #exponencial
imgG = np.zeros((rows,cols), dtype = "uint8") #exponencial
imgB = np.zeros((rows,cols), dtype = "uint8") #exponencial

intensity = 0.0;

for i in range(rows):
    for j in range(cols):
        intensity = img[i,j] / 255
        if intensity==0.0:
            #COLOR CELESTE
            imgR[i,j] = 100
            imgG[i,j] = 255
            imgB[i,j] = 255
        elif intensity <=0.25:
            #COLOR AZUL
            imgR[i,j] = 0
            imgG[i,j] = 0
            imgB[i,j] = 220
        elif intensity <=0.5:
            #COLOR VERDE
            imgR[i,j] = 0
            imgG[i,j] = 255
            imgB[i,j] = 0
        elif intensity <=0.75:
            #COLOR AMARILL0
            imgR[i,j] = 255
            imgG[i,j] = 255
            imgB[i,j] = 0
        elif intensity <=1:
            #COLOR ROJO
            imgR[i,j] = 255
            imgG[i,j] = 0
            imgB[i,j] = 0

img_rgb = cv.merge((imgB,imgG,imgR))
img_gray = cv.merge((img,img,img))

imgFinal = np.concatenate((img_gray, img_rgb), axis=1)

cv.imshow('COLOR MAP',imgFinal)
cv.waitKey(0)
cv.destroyAllWindows()
