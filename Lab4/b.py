
#---------------------------------------------------------------
#Librerias
import cv2 as cv
import numpy as np
from math import sqrt, e


def drawCircle(event,x,y,flags,param):
    global counter
    if event == cv.EVENT_LBUTTONDOWN:
        #print((x,y))
        if (x < center[0] and y < center[1] ):
            dx = abs(center[0]-x)
            dy = abs(center[1]-y)
            sx = center[0] + dx
            sy = center[1] + dy
            cv.circle(tmpfilter,(x,y),10,(0,0,0),-1)
            cv.circle(mag,(x,y),10,(0,0,0),-1)
            cv.circle(tmpfilter,(sx,sy),10,(0,0,0),-1)
            cv.circle(mag,(sx,sy),10,(0,0,0),-1)
        elif (x > center [0] and y > center[1]):
            dx = abs(center[0]-x)
            dy = abs(center[1]-y)
            sx = center[0] - dx
            sy = center[1] - dy
            cv.circle(tmpfilter,(x,y),10,(0,0,0),-1)
            cv.circle(mag,(x,y),10,(0,0,0),-1)
            cv.circle(tmpfilter,(sx,sy),10,(0,0,0),-1)
            cv.circle(mag,(sx,sy),10,(0,0,0),-1)
        elif (x <= center[0] and y > center[1] ):
            dx = abs(center[0]-x)
            dy = abs(center[1]-y)
            sx = center[0] + dx
            sy = center[1] - dy
            cv.circle(tmpfilter,(x,y),10,(0,0,0),-1)
            cv.circle(mag,(x,y),10,(0,0,0),-1)
            cv.circle(tmpfilter,(sx,sy),10,(0,0,0),-1)
            cv.circle(mag,(sx,sy),10,(0,0,0),-1)
        elif ( x > center[0] and y <= center[1]):
            dx = abs(center[0]-x)
            dy = abs(center[1]-y)
            sx = center[0] - dx
            sy = center[1] + dy
            cv.circle(tmpfilter,(x,y),10,(0,0,0),-1)
            cv.circle(mag,(x,y),10,(0,0,0),-1)
            cv.circle(tmpfilter,(sx,sy),10,(0,0,0),-1)
            cv.circle(mag,(sx,sy),10,(0,0,0),-1)
        elif (x == center[0] and y == center[1]):
            cv.circle(tmpfilter,center,10,(0,0,0),-1)
            cv.circle(mag,center,10,(0,0,0),-1)

        counter = counter + 1


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


def create_ButterworthBandRejectFilter(dft_Filter, D, n, W,centre):
    tmp = np.zeros((dft_Filter.shape[0] & -2,dft_Filter.shape[1] & -2), dtype='float32')

    for i in range(dft_Filter.shape[0] & -2):
        for j in range(dft_Filter.shape[1] & -2):
            radius = sqrt(pow((i - centre[0]), 2) + pow((j - centre[1]), 2))
            try:
                tmp[i,j] = 1 / (1 + pow((radius*W /(pow(radius,2)-pow(D,2))), (2 * n)))
            except:
                tmp[i,j] = 0

    dft_Filter = cv.merge((tmp,tmp))
    return dft_Filter
#Read Image
image = cv.imread("car.png" , 0)
#Assign Names to Windows
originalName = "Original image"
spectrumMagName = "Magnitude Image (log transformed)- spectrum"
maskName = "Mask Image"
filterName = "Filter Image"
#Create Windows
cv.namedWindow(originalName,cv.WINDOW_NORMAL)
cv.resizeWindow(originalName, 450,450)

cv.namedWindow(spectrumMagName,cv.WINDOW_NORMAL)
cv.resizeWindow(spectrumMagName, 450,450)
cv.setMouseCallback(spectrumMagName,drawCircle)
cv.namedWindow(maskName,cv.WINDOW_NORMAL)
cv.resizeWindow(maskName, 450,450)

cv.namedWindow(filterName,cv.WINDOW_NORMAL)
cv.resizeWindow(filterName, 450,450)

#Filter Output
filterOutput = np.array([])
padded = np.zeros(image.shape, dtype=image.dtype)
#Create DFT size
M = cv.getOptimalDFTSize(image.shape[0])
N = cv.getOptimalDFTSize(image.shape[1])
#Pad the image
padded = cv.copyMakeBorder(image, 0, M - image.shape[0], 0, N - image.shape[1], cv.BORDER_CONSTANT, value=0)
#Get the 2 planes
planes_0 = np.array(padded, dtype='float32')
planes_1 = np.zeros(padded.shape, dtype='float32')
complexImg = cv.merge((planes_0,planes_1))
#Create the discrete fourier transform
complexImg = cv.dft(complexImg)

#Create the filter
tmpfilter = np.ones((complexImg.shape[0] & -2,complexImg.shape[1] & -2), dtype='float32')
#Get real center
center = (tmpfilter.shape[1] // 2, tmpfilter.shape[0] // 2)
#Display orginal image
cv.imshow(originalName, image)

#Create flag
flag = 1
#Set counter to 0
counter = 0
#Create the display log pow
mag = create_spectrum_magnitude_display(complexImg, True)
#Display log pow
cv.imshow(spectrumMagName, mag)
#Click on the display to get noise position
while True:
    #Wait for click
    if(counter < flag):
        pass
    else:
        flag += 1
        if counter == 0:
            pass
        else:
            complexImg = shiftDFT(complexImg)
            filter=cv.merge((tmpfilter,tmpfilter))
            complexImg = cv.mulSpectrums(complexImg, filter, 0)
            complexImg = shiftDFT(complexImg)
            #Display log pow
            cv.imshow(spectrumMagName, mag)
        #Inverse Discrete Fourier Transform
        result = cv.idft(complexImg)
        #Transform DFT
        (myplanes_0,myplanes_1) = cv.split(result)
        result = cv.magnitude(myplanes_0,myplanes_1)
        result = cv.normalize(result,  result, 0, 1, cv.NORM_MINMAX)
        imageRes = result
        cv.imshow(maskName, imageRes)
        (planes_0,planes_1) = cv.split(filter)
        filterOutput = cv.normalize(planes_0, filterOutput, 0, 1, cv.NORM_MINMAX)
        cv.imshow(filterName,filterOutput)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
