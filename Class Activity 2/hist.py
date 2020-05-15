#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:50:34 2020

@author: lizeth
"""

import cv2 as cv
import numpy as np


def generateHistogram(img):
    histogram = np.zeros(256, dtype='int')
    for i in range(len(img)):
        for j in range(len(img[0])):
            histogram[img[i][j]]+=1
    return histogram

def displayHistogram(hist, name):
    tmp = list(hist)
    width = 500
    height = 500
    bin_width = int(round(width/256,0))
    histImg = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            histImg[i][j] = 255
    maximum = max(tmp)
    tmp = (tmp/maximum) * height
    for i in range(256):
        cv.line(histImg, (bin_width*i,height), (bin_width*i, int(height-tmp[i])), (0,0,0), 1)
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, width, height)
    cv.imshow(name, histImg)

def returnHistogram(hist):
    tmp = list(hist)
    width = 500
    height = 500
    bin_width = int(round(width/256,0))
    histImg = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            histImg[i][j] = 255
    maximum = max(tmp)
    tmp = (tmp/maximum) * height
    for i in range(256):
        cv.line(histImg, (bin_width*i,height), (bin_width*i, int(height-tmp[i])), (0,0,0), 1)
    return histImg

def cumHistogram(hist):
    cumHist = np.zeros(256, dtype='int')
    cumHist[0] = hist[0]
    for i in range(1,256):
        cumHist[i] = cumHist[i-1] + hist[i]
    return cumHist


#-------------------------------------------------------------------------------
#Lee la imagen a mejorar
img = cv.imread("coin1A.jpg", 0)
#Genera un histograma de la imagen a mejorar
hist = generateHistogram(img)

size = img.size

alpha = 255/size
PrRk = np.zeros(256)
for i in range(256):
    PrRk[i] = hist[i]/size

cumHist = cumHistogram(hist)

Sk = np.zeros(256, dtype='int')
for i in range(256):
    Sk[i] = int(round(cumHist[i]*alpha,0))

PsSk = np.zeros(256)
for i in range(256):
    PsSk[Sk[i]] += PrRk[i]

final = np.zeros(256, dtype='int')
for i in range(256):
    final[i] = int(round(PsSk[i]*255,0))

imgEqualizedHist = img.copy()
for i in range(len(imgEqualizedHist)):
    for j in range(len(imgEqualizedHist[0])):
        imgEqualizedHist[i][j] = Sk[img[i][j]]


histEqualized = generateHistogram(imgEqualizedHist)
histOriginal = returnHistogram(hist)
histFinal = returnHistogram(histEqualized)

#------------------------------------------------------------------
#Lee una imagen de calidad
img_ref = cv.imread("coin1.jpg", 0)
#Genera un histograma de la imagen de calidad
hist_ref = generateHistogram(img_ref)

size_ref = img_ref.size
alpha_ref = 255/size_ref
PrRk_ref = np.zeros(256)
for i in range(256):
    PrRk_ref[i] = hist_ref[i]/size_ref

cumHist_ref = cumHistogram(hist_ref)

Sk_ref = np.zeros(256, dtype='int')
for i in range(256):
    Sk_ref[i] = int(round(cumHist_ref[i]*alpha_ref,0))

PsSk_ref = np.zeros(256)
for i in range(256):
    PsSk_ref[Sk_ref[i]] += PrRk_ref[i]

final_ref = np.zeros(256, dtype='int')
for i in range(256):
    final_ref[i] = int(round(PsSk_ref[i]*255,0))

imgEqualizedHist_ref = img_ref.copy()
for i in range(len(imgEqualizedHist_ref)):
    for j in range(len(imgEqualizedHist_ref[0])):
        imgEqualizedHist_ref[i][j] = Sk_ref[img_ref[i][j]]


histEqualized_ref = generateHistogram(imgEqualizedHist_ref)
histOriginal_ref = returnHistogram(hist_ref)
histFinal_ref = returnHistogram(histEqualized_ref)

row1 = np.concatenate((cv.resize(img,(200,200),interpolation = cv.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv.resize(img_ref,(200,200),interpolation = cv.INTER_CUBIC)), axis = 1)
row2 = np.concatenate((cv.resize(histOriginal,(200,200),interpolation = cv.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv.resize(histOriginal_ref,(200,200),interpolation = cv.INTER_CUBIC)), axis = 1)
row3 = np.concatenate((cv.resize(histFinal,(200,200),interpolation = cv.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv.resize(histFinal_ref,(200,200),interpolation = cv.INTER_CUBIC)), axis = 1)

finalImg = np.concatenate((row1,row2,row3), axis=0)
cv.imshow("Imgs Leidas/Hist Originales/Hist Equalizados", finalImg)


#------------------------------------------------------------------------
#Genera imagen mejorada

prob_cum_hist=cumHist/size
prob_cum_hist_ref=cumHist_ref/size_ref

K=256
new_values=np.zeros((K))

for a in np.arange(K):
    j=K-1
    while True:
        new_values[a]=j
        j=j-1
        if j<0 or prob_cum_hist[a]>prob_cum_hist_ref[j]:
            break

for i in np.arange(size):
        a=img.item(i)
        b=new_values[a]
        img.itemset((i),b)


hist_m = generateHistogram(img)
size_m = img.size
alpha_m = 255/size_m
PrRk_m = np.zeros(256)
for i in range(256):
    PrRk_m[i] = hist_m[i]/size_m

cumHist_m = cumHistogram(hist_m)

Sk_m = np.zeros(256, dtype='int')
for i in range(256):
    Sk_m[i] = int(round(cumHist_m[i]*alpha_m,0))

PsSk_m = np.zeros(256)
for i in range(256):
    PsSk_m[Sk_m[i]] += PrRk_m[i]

final_m = np.zeros(256, dtype='int')
for i in range(256):
    final_m[i] = int(round(PsSk_m[i]*255,0))

imgEqualizedHist_m = img.copy()
for i in range(len(imgEqualizedHist_m)):
    for j in range(len(imgEqualizedHist_m[0])):
        imgEqualizedHist_m[i][j] = Sk_m[img[i][j]]


histEqualized_m = generateHistogram(imgEqualizedHist_m)
histOriginal_m = returnHistogram(hist_m)
histFinal_m = returnHistogram(histEqualized_m)


row1 = np.concatenate((cv.resize(img,(200,200),interpolation = cv.INTER_CUBIC), np.zeros((200,10), dtype='uint8') ), axis = 1)
row2 = np.concatenate((cv.resize(histOriginal_m,(200,200),interpolation = cv.INTER_CUBIC), np.zeros((200,10), dtype='uint8') ), axis = 1)

finalImg2 = np.concatenate((row1,row2), axis=0)

#Muestra la imagen mejorada con su histograma
cv.imshow('Mejorada-Histograma',finalImg2)

cv.waitKey(0)
cv.destroyAllWindows()
