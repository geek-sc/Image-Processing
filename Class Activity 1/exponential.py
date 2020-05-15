#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:03:53 2020

@author: lizeth
"""

import numpy as np
import cv2 as cv

#Load an color image in grayscale
img= cv.imread('Lab1B.3.png',0)
#Obtain image dimensions
rows,cols=img.shape
#Initialize variables
imgResult=np.zeros((rows,cols),dtype="uint8")

#potencia
# S = c * [f(x,y)]^n   si n es 0-1 se aclara, si es mayor a 1 se obscurece
for i in range(rows):
    for j in range(cols):
        imgResult[i,j] = np.power(img[i,j],1.2)


image = cv.hconcat([img,imgResult])
cv.imshow('Original / Exponential',image)

cv.waitKey(0)
cv.destroyAllWindows()
