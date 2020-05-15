#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:49:11 2020

@author: lizeth
"""
import cv2
import numpy as np


#Load an color image in grayscale
img = cv2.imread('cameraman.jpg',0)

#Obtain image dimensions
rows,cols = img.shape

#Initialize variables
imgResult=np.zeros((rows,cols),dtype="uint8")


#Calculate the inverse or negative image s(x,y)=255-r
for i in range(rows):
    for j in range(cols):
        imgResult[i,j] = 256-img[i,j]



#Separated images
cv2.imshow('image',img)
cv2.imshow('imageResult',imgResult)

#Concatenated images
numpy_vertical = np.vstack((img, imgResult))
cv2.imshow('Original / Inverse ',numpy_vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()
