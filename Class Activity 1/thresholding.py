#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:51:10 2020

@author: lizeth
"""

import numpy as np
import cv2

#Load an color image in grayscale
img= cv2.imread('cameraman.jpg',0)
#Obtain image dimensions
rows,cols=img.shape
#Initialize variables
imgResult=np.zeros((rows,cols),dtype="uint8")
th = 120

#Thresholding
for i in range(rows):
    for j in range(cols):
        if img[i,j]>th:
            imgResult[i,j]=255

#show images
#cv2.imshow("Image Result",imgResult)
#cv2.imshow("Original image",img)
image = cv2.hconcat([img,imgResult])
cv2.imshow('Original / Thresholding',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
