#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 07:22:27 2020

@author: lizeth
"""


import numpy as np
import cv2
# Read the image in greyscale
img = cv2.imread('cameraman.jpg',0)

#Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

# We have a list of strings where each string represents binary pixel value. To extract bit planes we need to iterate over the strings and store the characters corresponding to bit planes into lists.
# Multiply with 2^(n-1) and reshape to reconstruct the bit image.
eight_bit_img = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
seven_bit_img = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
six_bit_img = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(img.shape[0],img.shape[1])
five_bit_img = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(img.shape[0],img.shape[1])
four_bit_img = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(img.shape[0],img.shape[1])
three_bit_img = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(img.shape[0],img.shape[1])
two_bit_img = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(img.shape[0],img.shape[1])
one_bit_img = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(img.shape[0],img.shape[1])

# Adding bit planes starting with the eight bit and so on in  a decreasing way
new_img1 = eight_bit_img+seven_bit_img+six_bit_img+five_bit_img+four_bit_img+three_bit_img+two_bit_img+one_bit_img
new_img2 = eight_bit_img+seven_bit_img+six_bit_img+five_bit_img+four_bit_img+three_bit_img+two_bit_img
new_img3 = eight_bit_img+seven_bit_img+six_bit_img+five_bit_img+four_bit_img+three_bit_img
new_img4 = eight_bit_img+seven_bit_img+six_bit_img+five_bit_img+four_bit_img
new_img5 = eight_bit_img+seven_bit_img+six_bit_img+five_bit_img
new_img6 = eight_bit_img+seven_bit_img+six_bit_img
new_img7 = eight_bit_img+seven_bit_img
new_img8 = eight_bit_img

# Combining bit planes starting with the one bit and so on in  an increasing way
new_img11 = one_bit_img+two_bit_img+three_bit_img+four_bit_img+five_bit_img+six_bit_img+seven_bit_img+eight_bit_img
new_img21 = one_bit_img+two_bit_img+three_bit_img+four_bit_img+five_bit_img+six_bit_img+seven_bit_img
new_img31 = one_bit_img+two_bit_img+three_bit_img+four_bit_img+five_bit_img+six_bit_img
new_img41 = one_bit_img+two_bit_img+three_bit_img+four_bit_img+five_bit_img
new_img51= one_bit_img+two_bit_img+three_bit_img+four_bit_img
new_img61 = one_bit_img+two_bit_img+three_bit_img
new_img71 = one_bit_img+two_bit_img
new_img81 = one_bit_img

# Vertically concatenate
#finalr = cv2.hconcat([new_img1,new_img2,new_img3,new_img4])
#finalv =cv2.hconcat([new_img5,new_img6,new_img7,new_img8])

finalr = cv2.hconcat([new_img8,new_img7,new_img6,new_img5])
finalv = cv2.hconcat([new_img4,new_img3,new_img2,new_img1])

finalr1 = cv2.hconcat([new_img11,new_img21,new_img31,new_img41])
finalv1 =cv2.hconcat([new_img51,new_img61,new_img71,new_img81])


# Vertically concatenate
final = cv2.vconcat([finalr,finalv])
final1 = cv2.vconcat([finalr1,finalv1])


# Display the image
#cv2.imshow('c',new_img)
cv2.imshow(' Adding bit planes starting with the eight bit and so on in  a decreasing way',final)
cv2.imshow('Adding bit planes  starting with the one bit and so on in  an increasing way',final1)

cv2.waitKey(0)
