#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:08:30 2020

@author: lizeth
"""

import numpy as np
import cv2
import math

#Load the image
img= cv2.imread('cameraman.jpg',0)
rows,cols=img.shape
imgResult=np.zeros((rows,cols,1),dtype="uint8") #uint8 unsigned integer

for i in range(rows):
    for j in range(cols):
        if img[i,j]<70:
            imgResult[i,j]=img[i,j]**1.2 # Power low> then, black color
        elif (img[i,j]>=70 and img[i][j] <150):
            imgResult[i,j]=255-img[i,j] #inverse image
        elif (img[i,j]>=150 and img[i][j] <200):
            imgResult[i,j]=img[i,j]**0.6  #Power low
        elif img[i,j]>=200:
            imgResult[i,j]=math.log(img[i,j]+1) # log

cv2.imshow("Piecewise function",imgResult)
cv2.imshow("Original Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
