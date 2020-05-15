'''
Lab 5: Band Reject Filter
'''

#------------------------------------------------------------------------
#Librerias
import numpy as np
import cv2 as cv
import random
from math import sqrt, e

#-------------------------------------------------------------------------
#funcion para fourier
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

#----------------------------------------------------------------------
#Funcion para crear el espectro
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

#-----------------------------------------------------------------------------------------
def create_BandRejectLowpassFilter(dft_Filter, D, n, W):
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


def create_BandRejectHighpassFilter(dft_Filter, D, n, W):
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

#----------------------------------------------------------------------------------
def nothing(x):
    pass

#----------------------------------------------------------------------------
image = cv.imread("nasa.png" , 0) #Lee una imagen en escala gris

#Loop para recorrer la imagen y leer los bytes
image = cv.resize(image,(250,250))
org = np.zeros(image.shape, dtype="float64")
for i in range(250):
    for j in range(250):
        org[i,j] = image[i,j]/255

#Definiciones de arrays para los filtros y el padding
filterOutput = np.array([])
filterOutput2 = np.array([])
padded3 = np.zeros(image.shape, dtype=image.dtype)

#------------------------------------------------------------------
# Crea las barras para radio, orden y ancho

cv.namedWindow("Band Reject Filter")
cv.createTrackbar('Radius','Band Reject Filter',1,250,nothing)
cv.createTrackbar('Order','Band Reject Filter',1,20,nothing)
cv.createTrackbar('Width','Band Reject Filter',1,100,nothing)

#-----------------------------------------------------------------------
# Main loop

while(1):

    #Obtiene las barras creadas para raiod, orden y ancho
    radius = cv.getTrackbarPos('Radius',"Band Reject Filter")
    order = cv.getTrackbarPos('Order',"Band Reject Filter")
    width = cv.getTrackbarPos('Width',"Band Reject Filter")
    radius2 = radius + width

    M = cv.getOptimalDFTSize(image.shape[0])
    N = cv.getOptimalDFTSize(image.shape[1])

    padded3 = cv.copyMakeBorder(image, 0, M - image.shape[0], 0, N - image.shape[1], cv.BORDER_CONSTANT, value=0)

    planes_0 = np.array(padded3, dtype='float32')
    planes_1 = np.zeros(padded3.shape, dtype='float32')
    planes2_0 = np.array(padded3, dtype='float32')
    planes2_1 = np.zeros(padded3.shape, dtype='float32')

    complexImg = cv.merge((planes_0,planes_1))
    complexImg = cv.dft(complexImg)

    complexImg2 = cv.merge((planes2_0,planes2_1))
    complexImg2 = cv.dft(complexImg2)


    filter1 = complexImg.copy()
    filter1 = create_BandRejectLowpassFilter(filter1, radius, order, width)
    filter2 = complexImg2.copy()
    filter2 = create_BandRejectHighpassFilter(filter2, radius2, order, width)
    filter3 = complexImg.copy()
    filter3 = filter1+filter2

    complexImg = shiftDFT(complexImg)
    complexImg = cv.mulSpectrums(complexImg, filter3, 0)
    complexImg = shiftDFT(complexImg)


    mag = create_spectrum_magnitude_display(complexImg, True)

    result = cv.idft(complexImg)

    (myplanes_0,myplanes_1) = cv.split(result)
    result = cv.magnitude(myplanes_0,myplanes_1)
    result = cv.normalize(result,  result, 0, 1, cv.NORM_MINMAX)
    imageRes = result

    (planes_0,planes_1) = cv.split(filter3)
    filterOutput = cv.normalize(planes_0, filterOutput, 0, 1, cv.NORM_MINMAX)

    finalfreq = np.hstack((org,mag,imageRes))
    a = np.hstack((org,mag))
    b = np.hstack((filterOutput,imageRes))
    c = np.vstack((a,b))

    cv.imshow('Band Reject Filter',c)

    k = cv.waitKey(1) & 0XFF
    if k == 27:
        break
cv.destroyAllWindows()
