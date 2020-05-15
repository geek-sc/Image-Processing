#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:40:39 2020

@author: lizeth
"""

import cv2 as cv
import numpy as np
#import math
#-------------------------------------------------------------------

def rgb2ycbcr(im):
    #Recibe como parametro unaa imagen a COLOR
    #Define un array vacio para ycbcr
    ycbcr = np.empty_like(im)

    #extrae los canales r,g,b
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    # Y
    #ycbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    #ycbcr[:,:,1] = 128 - .168736 * r - .331264 * g + .500 * b
    # Cr
    #ycbcr[:,:,2] = 128 + .5 * r - .418688 * g - .081312 * b

    ycbcr[:,:,0] = 0.257 * r + 0.504 * g + (0.098 * b +16)
    # Cb
    ycbcr[:,:,1] = - 0.148 * r - 0.291 * g +  (0.439 * b +128)
    # Cr
    ycbcr[:,:,2] = 0.439 * r - 0.368 * g - (0.071 * b +128)

    return np.uint8(ycbcr)

def ycbcr2rgb(im):

    rgb=np.empty_like(im)
    y = im[:,:,0]
    cb = im[:,:,1]
    cr = im[:,:,2]

    #rgb[:,:,0]=  y + 0 * (cb - 128) +  1.402 * (cr - 128)
    #rgb[:,:,1]=  y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    #rgb[:,:,2]=  y + 1.772 *(cb - 128) +  0 * (cr - 128)

    rgb[:,:,0]=  y + 1.402 * (cr - 128)
    rgb[:,:,1]=  y - 0.344 * (cb - 128) - 0.714 * (cr - 128)
    rgb[:,:,2]=  y + 1.772 *(cb - 128)
    return np.uint8(rgb)



def rgb_to_cmy(im):
    #Recibe como parametro unaa imagen a COLOR
    #Define un array vacio para cmy
    cmy = np.empty_like(im)

    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    cmy[:,:,0] = (1 - r)
    cmy[:,:,1] = (1 - g)
    cmy[:,:,2] = (1 - b)

    return np.uint8(cmy)

def cmy_to_rgb(im):
    rgb = np.empty_like(im)

    c = im[:,:,0]
    m = im[:,:,1]
    y = im[:,:,2]

    rgb[:,:,0] = (1 - c)
    rgb[:,:,1] = (1 - m)
    rgb[:,:,2] = (1 - y)
    return np.uint8(rgb)


#-------------------------------------------------------------------------------------------------

# Import color picture
img = cv.imread('peppers256.png', 1)

#----------------------------------MODELO 1 YCBCR--------------------------------------------------------
img_ycbcr = rgb2ycbcr(img)
img_rgb=ycbcr2rgb(img_ycbcr)
model1 = cv.hconcat([img,img_ycbcr,img_rgb])
cv.imshow('ORIGINAL / RGB -> YCBCR / YCBCR -> RGB',model1)

#----------------------------------MODELO 2 CMY--------------------------------------------------------
img_cmy = rgb_to_cmy(img)
img_bgr = cmy_to_rgb(img_cmy)
model2 = cv.hconcat([img,img_cmy,img_bgr])
cv.imshow('ORIGINAL / RGB -> CMY / CMY -> RGB',model2)

cv.waitKey(0)
cv.destroyAllWindows()
