
import numpy as np
import cv2 as cv

#histogram generation function
def generateHistogram(img):
    histogram = np.zeros(256, dtype='int')
    for i in range(len(img)):
        for j in range(len(img[0])):
            histogram[img[i][j]]+=1
    return histogram

#turn histogram as an 'image' for showing function
def returnHistogram(hist):
    tmp = list(hist)
    width = 500
    height = 500
    bin_width = int(round(width/256,0))
    histImg = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            histImg[i][j] = 255 #white
    maximum = max(tmp)
    tmp = (tmp/maximum) * height
    for i in range(256):
        cv.line(histImg, (bin_width*i,height), (bin_width*i, int(height-tmp[i])), (0,0,0), 1)
    return histImg

#read image
img = cv.imread('imgs/lena.jpg', 0)

#create copy of image for image + noise average
imgavg = img

#obtain image dimensions
rows,cols = img.shape

#create noise variable (array)
Noise = np.zeros((rows,cols), dtype='uint8')

#create error variable for error image
imgerror = np.zeros((rows,cols), dtype='uint8')
#medium
m = 20
#
#Standard deviation of Gaussian distribution
sd = 15

#counter variable
count = 1

k=6
for i in range (k):
    cv.randn(Noise, m, sd) #create noise
    newimg = np.zeros((rows,cols), dtype='uint8')
    newimg = img + Noise #worsen the image
    count = count + 1 #total number of images
    imgavg = (imgavg + Noise) #/count, how to divide?

    imgerror = cv.absdiff(imgavg, img)
    hist = generateHistogram(imgerror)
    histimg = returnHistogram(hist)
    

    row1 = np.concatenate((cv.resize(newimg,(150,150),interpolation = cv.INTER_CUBIC), np.zeros((150,0), dtype='uint8')), axis = 1)
    row2 = np.concatenate((cv.resize(imgavg,(150,150),interpolation = cv.INTER_CUBIC), np.zeros((150,0), dtype='uint8')), axis = 1)
    row3 = np.concatenate((cv.resize(histimg,(150,150),interpolation = cv.INTER_CUBIC), np.zeros((150,0), dtype='uint8')), axis = 1)
    row4 = np.concatenate((cv.resize(imgerror,(150,150),interpolation = cv.INTER_CUBIC), np.zeros((150,0), dtype='uint8')), axis = 1)
    finalImg = np.concatenate((row1,row2,row3,row4), axis=0)
    cv.imshow("Noise,avg,hist,imgerror", finalImg)

    cv.waitKey(0)
