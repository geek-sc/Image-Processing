'''
Lab 6 : Wiener Filter
'''
#----------------------------------------
#Librerias
import numpy as np
import cv2 as cv
from math import sqrt, e

#---------------------------------------------------------------#
#                               FUNCIONES                       #
#---------------------------------------------------------------#

def nothing(x):
    pass
#----------------------------------------------------------------
#Funcion para la transformada de fourier
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
#Funcion para obtener PSF
def getPSF(PSF, R):
    '''
    R: matriz de correlacion entre dos vectores x, y
    PSF: P: mejor matriz
         S: espectro de potencia
         F: estimacion
    '''
    PSF = np.ones((R, R)) / 25
    PSF = cv.merge((PSF,PSF))
    return PSF*255
#-------------------------------------------------------------
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
#---------------------------------------------------------------
#Funcion para filtro de frecuencia
def create_Frequency2DFiltering(complexImg, filter):
    #complexImg = cv.dft(complexImg)#Fourier
    complexImg = shiftDFT(complexImg)
    complexImg = cv.mulSpectrums(complexImg, filter, 0)

    complexImg = shiftDFT(complexImg)#Inverse Fourier
    m = create_spectrum_magnitude_display(complexImg,True)
    result = cv.idft(complexImg)

    (myplanes_0,myplanes_1) = cv.split(result)
    result = cv.magnitude(myplanes_0,myplanes_1)
    result = cv.normalize(result,  result, 0, 1, cv.NORM_MINMAX)
    imageRes = result
    return imageRes,m;
#------------------------------------------------------------------
#Funcion para Filtro Inverso del dft
def create_InverseFilter(dft_Filter):
    return 1/dft_Filter

#------------------------------------------------------------------
#Funcion para filtro de Butterworth
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
#---------------------------------------------------------------
#Funcion para Filtro de Wiener
def create_WienerFilter(H, SNR1):
    '''
    SNR: Signal to Noise Ratio en el dominio de la frecuencia
    '''
    (planes_0,planes_1) = cv.split(H)
    denom=np.power(planes_0, 2)
    denom=denom + SNR1;
    planes_0= planes_0/denom
    #inverse = np.power(1/(planes_0.copy()+0.000001), 2) + SNR1
    #planes_0= planes_0*inverse

    dft_Filter = cv.merge((planes_0,planes_0))
    return dft_Filter
#------------------------------------------------------------------

#funciones para concatenar las imagenes
def vconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv.INTER_CUBIC)

#---------------------------------------------------------------
image = cv.imread("lena.jpg" , 0)
#image = average_filter(image,3)

image = cv.resize(image,(300,300))

filterOutput = np.array([])
wienerfilterOutput= np.array([])
padded = np.zeros(image.shape, dtype=image.dtype)

#------------------------------------------------------------------
# Crea las barras para radio, orden y ancho

cv.namedWindow("Wiener")
cv.createTrackbar('Radius','Wiener',1,20,nothing)
cv.createTrackbar('Order','Wiener',1,3,nothing)
cv.createTrackbar('Width','Wiener',1,10,nothing)
#--------------------------------------------------
#Main

while(1):
    #Obtiene las barras creadas para radio, orden y ancho

    radius_w = cv.getTrackbarPos('Radius',"Wiener")
    order_w = cv.getTrackbarPos('Order',"Wiener")
    width_w = cv.getTrackbarPos('Width',"Wiener")
    fradius_w = radius_w + width_w

    M = cv.getOptimalDFTSize(image.shape[0])
    N = cv.getOptimalDFTSize(image.shape[1])

    padded = cv.copyMakeBorder(image, 0, M - image.shape[0], 0, N - image.shape[1], cv.BORDER_CONSTANT, value=0)
    planes_0 = np.array(padded, dtype='float32')
    planes_1 = np.zeros(padded.shape, dtype='float32')
    complexImg = cv.merge((planes_0,planes_1))
    complexImg = cv.dft(complexImg)

    filter = complexImg.copy()
    filter = create_ButterworthLowpassFilter(filter, fradius_w, order_w, width_w)
    wiener_filter = create_WienerFilter(filter.copy(), 0)


    imageRes,mag=create_Frequency2DFiltering(complexImg, filter)
    img_w,g = create_Frequency2DFiltering(complexImg,wiener_filter)

    #mag = create_spectrum_magnitude_display(cv.dft(complexImg), True)


    (planes_0,planes_1) = cv.split(filter)
    filterOutput = cv.normalize(planes_0, filterOutput, 0, 1, cv.NORM_MINMAX)

    (planes_0,planes_1) = cv.split(wiener_filter)
    wienerfilterOutput = cv.normalize(planes_0, wienerfilterOutput, 0, 1, cv.NORM_MINMAX)

    row1 = cv.hconcat([imageRes,wienerfilterOutput])
    row2 = cv.hconcat([filterOutput,img_w])
    final = cv.vconcat([row1,row2])
    #final = cv.hconcat([imageRes,img_w])
    #cv.imshow('Wiener', final)

    final = concat_tile_resize([[filterOutput,wienerfilterOutput,mag],
                                [imageRes,img_w]])
    cv.imshow('Wiener', final)
    k = cv.waitKey(1) & 0XFF
    if k == 27:
        break
cv.destroyAllWindows()
