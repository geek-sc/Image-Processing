#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:04:06 2020

@author: lizeth
"""

import numpy as np
import cv2 as cv


#Load an color image in grayscale
img= cv.imread('lena.jpg',0)
#Obtain image dimensions
rows,cols=img.shape
#Initialize variables
imgResult=np.zeros((rows,cols),dtype="uint8")

#Logaritmic image
# S = c * log (1 + f(x,y))
for i in range(rows):
    for j in range(cols):
        imgResult[i,j] = 200*np.log10(1+img[i,j])


image = cv.hconcat([img,imgResult])
cv.imshow('Original / Logaritmo',image)

cv.waitKey(0)
cv.destroyAllWindows()
