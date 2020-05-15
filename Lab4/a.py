#!/usr/bin/env python
# coding: utf-8

#--------------------------------------------------------------
import numpy as np
import cv2 as cv
from math import sqrt, e

#---------------------------------------------------------------
#Transformada de Fourier
def shiftDFT(fImage):
    fImage = fImage[0:fImage.shape[0] & -2, 0:fImage.shape[1] & -2]

    cx = fImage.shape[1] // 2
    cy = fImage.shape[0] // 2

    q0 = fImage[0:cy, 0:cx]
    q1 = fImage[0:cy, cx:cx+cx]
    q2 = fImage[cy:cy+cy, 0:cx]
    q3 = fImage[cy:cy+cy, cx:cx+cx]

    tmp = np.zeros(q0.shape, dtype=q0.dtype)
    np.copyto(tmp,q0)
    np.copyto(q0,q3)
    np.copyto(q3,tmp)

    np.copyto(tmp,q1)
    np.copyto(q1,q2)
    np.copyto(q2,tmp)
    return fImage

#----------------------------------------------------------------
#Funcion para generar un espectro
def create_spectrum_magnitude_display(complexImg, rearrange):
    (planes_0, planes_1) = cv.split(complexImg)
    planes_0 = cv.magnitude(planes_0, planes_1)

    mag = planes_0.copy()
    mag += 1
    mag = cv.log(mag)

    if (rearrange):
        shiftDFT(mag);

    mag = cv.normalize(mag,  mag, 0, 1, cv.NORM_MINMAX)
    return mag

#-----------------------------------------------------------------------
#Filtro de paso bajo
def create_ButterworthLowpassFilter(dft_Filter, D, n, W):
    tmp = np.zeros((dft_Filter.shape[0] & -2,dft_Filter.shape[1] & -2), dtype='float32')

    centre = ((dft_Filter.shape[0] & -2) // 2, (dft_Filter.shape[1] & -2) // 2)

    for i in range(dft_Filter.shape[0] & -2):
        for j in range(dft_Filter.shape[1] & -2):
            radius = sqrt(pow((i - centre[0]), 2) + pow((j - centre[1]), 2))
            try:
                tmp[i,j] = 1 / (1 + pow((radius /  D), (2 * n)))
            except:
                tmp[i,j] = 0

    dft_Filter = cv.merge((tmp,tmp))
    return dft_Filter

#---------------------------------------------------------------------------
#Filtro de paso alto
def create_ButterworthHighpassFilter(dft_Filter, D, n, W):
    tmp = np.zeros((dft_Filter.shape[0] & -2,dft_Filter.shape[1] & -2), dtype='float32')

    centre = ((dft_Filter.shape[0] & -2) // 2, (dft_Filter.shape[1] & -2) // 2)

    for i in range(dft_Filter.shape[0] & -2):
        for j in range(dft_Filter.shape[1] & -2):
            radius = sqrt(pow((i - centre[0]), 2) + pow((j - centre[1]), 2))
            try:
                tmp[i,j] = 1 / (1 + pow((D /  radius), (2 * n)))
            except:
                tmp[i,j] = 0

    dft_Filter = cv.merge((tmp,tmp))
    return dft_Filter

#--------------------------------------------------------------------------------
def nothing(x):
    pass

#------------------------------------------------------------------------------
#Lee una imagen a escala de gris
image = cv.imread("lena.jpg" , 0)
rows,cols  = image.shape

#--------------------------------------------------------------------
#Matrices a usar
filterOutput = np.array([])
filterOutput2 = np.array([])

padded = np.zeros(image.shape, dtype=image.dtype)
org = np.zeros(image.shape, dtype="float64")

for i in range(rows):
    for j in range(cols):
        org[i,j] = image[i,j]/255

#---------------------------------------------------------
#Transformaciones espaciales con kernel
KernelScharrH = np.array(((-1,-2,-1),(0,0,0),(1,2,1)))
KernelScharrV = np.array(((-1,0,1),(-2,0,2),(-1,0,1)))
KernelGauss = np.array(((1,2,1),(2,4,2),(1,2,1)))
KernelGauss = KernelGauss * (1/16)
imgScharrH = np.zeros((rows,cols), dtype = "uint8")
imgScharrV = np.zeros((rows,cols), dtype = "uint8")
imgScharrResult = np.zeros((rows,cols), dtype = "uint8")

#-------------------------------------------------------------
#Aplica las transformaciones

imgScharrH = cv.filter2D(org,-1,KernelScharrH)
imgScharrV = cv.filter2D(org,-1,KernelScharrV)
imgGaus = cv.filter2D(org,-1,KernelGauss)
imgScharrResult = imgScharrH + imgScharrV

#-----------------------------------------------------------

separador = np.zeros((rows,4), dtype = "uint8")
for i in range(rows):
    for j in range(4):
        separador[i,j] = 0
separador2 = np.zeros((rows,4), dtype = "uint8")
for i in range(rows):
    for j in range(4):
        separador2[i,j] = 255

#-----------------------------------------------------------
#Creamos las trackbars
cv.namedWindow("Low Pass Filter")
cv.namedWindow("High Pass Filter")
cv.createTrackbar('Radius','Low Pass Filter',1,100,nothing)
cv.createTrackbar('Order','Low Pass Filter',1,10,nothing)
cv.createTrackbar('Radius','High Pass Filter',1,100,nothing)
cv.createTrackbar('Order','High Pass Filter',1,10,nothing)

#--------------------------------------------------------------
#Loop para aplicar los filtros de acuerdo al orden y radio
while(1):
    radius = cv.getTrackbarPos('Radius',"Low Pass Filter")
    order = cv.getTrackbarPos('Order',"Low Pass Filter")
    radius2 = cv.getTrackbarPos('Radius',"High Pass Filter")
    order2 = cv.getTrackbarPos('Order',"High Pass Filter")

    width = 3
    M = cv.getOptimalDFTSize(image.shape[0])
    N = cv.getOptimalDFTSize(image.shape[1])

    padded = cv.copyMakeBorder(image, 0, M - image.shape[0], 0, N - image.shape[1], cv.BORDER_CONSTANT, value=0)

    planes_0 = np.array(padded, dtype='float32')
    planes_1 = np.zeros(padded.shape, dtype='float32')
    planes2_0 = np.array(padded, dtype='float32')
    planes2_1 = np.zeros(padded.shape, dtype='float32')

    complexImg = cv.merge((planes_0,planes_1))
    complexImg = cv.dft(complexImg)

    complexImg2 = cv.merge((planes2_0,planes2_1))
    complexImg2 = cv.dft(complexImg2)


    filter1 = complexImg.copy()
    filter1 = create_ButterworthLowpassFilter(filter1, radius, order, width)
    filter2 = complexImg2.copy()
    filter2 = create_ButterworthHighpassFilter(filter2, radius2, order2, width)


    complexImg = shiftDFT(complexImg)
    complexImg = cv.mulSpectrums(complexImg, filter1, 0)
    complexImg = shiftDFT(complexImg)

    complexImg2 = shiftDFT(complexImg2)
    complexImg2 = cv.mulSpectrums(complexImg2, filter2, 0)
    complexImg2 = shiftDFT(complexImg2)


    mag = create_spectrum_magnitude_display(complexImg, True)
    mag2 = create_spectrum_magnitude_display(complexImg2, True)


    result = cv.idft(complexImg)
    result2 = cv.idft(complexImg2)

    (myplanes_0,myplanes_1) = cv.split(result)
    result = cv.magnitude(myplanes_0,myplanes_1)
    result = cv.normalize(result,  result, 0, 1, cv.NORM_MINMAX)
    imageRes = result

    (myplanes2_0,myplanes2_1) = cv.split(result2)
    result2 = cv.magnitude(myplanes2_0,myplanes2_1)
    result2 = cv.normalize(result2,  result2, 0, 1, cv.NORM_MINMAX)
    imageRes2 = result2


    (planes_0,planes_1) = cv.split(filter1)
    filterOutput = cv.normalize(planes_0, filterOutput, 0, 1, cv.NORM_MINMAX)

    (planes2_0,planes2_1) = cv.split(filter2)
    filterOutput2 = cv.normalize(planes2_0, filterOutput2, 0, 1, cv.NORM_MINMAX)

    final = np.hstack((org,separador2,mag,separador2,imageRes,separador2,imgGaus))
    final2 = np.hstack((org,separador2,mag2,separador2,imageRes2,separador2,imgScharrResult))
    imgfinal3 =np.vstack((final,final2))
    cv.imshow('Low Pass Filter and High Pass Filter',imgfinal3)

    cv.imshow("Low Pass Filter", filterOutput)
    cv.imshow("High Pass Filter", filterOutput2)

    k = cv.waitKey(1) & 0XFF
    if k == 27:
        break

cv.destroyAllWindows()
