#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:34:54 2020

@author: lizeth
"""

import cv2
import numpy as np


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

img = cv2.imread('t3.jpg',0)
fil,col = img.shape
RGB = np.zeros((fil,col,3),dtype = 'uint8')

dvi = 100
mapc = np.zeros((500, 100, 3), dtype = "uint8")
R1 = np.zeros((500, 100, 1), dtype = "uint8")
G1 = np.zeros((500, 100, 1), dtype = "uint8")
B1 = np.zeros((500, 100, 1), dtype = "uint8")
for i in range(100):
    for j in range(100):
        R1[i,j] = 0
        G1[i,j] = ((0)*(i/dvi))
        B1[i,j] = ((0)*(i/dvi)) + 128

        R1[i+dvi,j] = 0
        G1[i+dvi,j] = ((255)*(i/dvi))
        B1[i+dvi,j] = ((-128)*(i/dvi)) + 128

        R1[i+(dvi*2),j] = 255*(i/dvi)
        G1[i+(dvi*2),j] = ((0)*(i/dvi)) +255
        B1[i+(dvi*2),j] = ((0)*(i/dvi))

        R1[i+(dvi*3),j] = ((0)*(i/dvi)) + 255
        G1[i+(dvi*3),j] = ((128-255)*(i/dvi)) +255
        B1[i+(dvi*3),j] = ((0)*(i/dvi))

        R1[i+(dvi*4),j] = ((0)*(i/dvi)) +255
        G1[i+(dvi*4),j] = ((-128)*(i/dvi)) +128
        B1[i+(dvi*4),j] = 0

map = cv2.merge((B1,G1,R1))
#cv2.imshow('mapa',mapc)



def inte(intensity):
    if intensity<0:
        R = 0
        G = 0
        B = 128
    elif (intensity > 0) and (intensity<=0.25):
        fraction = intensity
        R =  (0) * fraction
        G =  (255) * fraction
        B =  (-128) * fraction + 128

    elif(intensity>0.25) and (intensity<= 0.5):
        fraction = intensity
        R =  (255) * fraction
        G =  (255-255) * fraction +255
        B =  (0) * fraction
    elif(intensity>0.5) and (intensity<= 0.75):
        fraction = intensity/(0.75-0.5)
        R =  (255-255) * fraction + 255
        G =  (128-255) * fraction + 255
        B =  (0) * fraction

    elif (intensity>0.75) and (intensity<=1) :
        fraction = intensity
        R =  (255-255) * fraction + 255
        G =  (-128) * fraction + 128
        B =  (- 0) * fraction

    else:
        R = 255
        G = 0
        B = 0
    return R,G,B

intensity = 0.0
for i in range(fil):
    for j in range(col):
        intensity = img[i][j]/255.0
        r,g,b = inte(intensity)
        RGB[i][j][2] = r
        RGB[i][j][1] = g
        RGB[i][j][0] = b


image = cv2.merge((RGB[:,:,0],RGB[:,:,1],RGB[:,:,2]))
r = image[:,:,2]
cv2.imshow('r',r)
cv2.imwrite('pol.jpg',r)

img_gray = cv2.merge((img,img,img))
imgFinal = np.concatenate((img_gray, image), axis=1)
#cv2.imshow('INTERPOLATION',imgFinal)
#cv2.imshow('Map',map)

im_h_resize = hconcat_resize_min([img_gray,map, image])
cv2.imshow('Interpolation',im_h_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
