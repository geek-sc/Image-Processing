"""
Lab 2 Activity a
"""

#Librerias  necesarias
import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt


#Funcion para generar un histograma
#Toma como entrada una imagen para generar un histograma
def generateHistogram(img):
    histogram = np.zeros(256, dtype='int')
    for i in range(len(img)):
        for j in range(len(img[0])):
            histogram[img[i][j]]+=1
    return histogram


#Funcion que muestra el histograma
#Como parametro un histograma y el nombre
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

#Funcion que retorna un histograma de una imagen
#Toma como entrada un histograma
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

#Funcion de histograma acumulado
def cumHistogram(hist):
    cumHist = np.zeros(256, dtype='int')
    cumHist[0] = hist[0]
    for i in range(1,256):
        cumHist[i] = cumHist[i-1] + hist[i]
    return cumHist


#Lee una imagen
img = cv.imread("Lab3/coin1A.jpg", 0)
hist = generateHistogram(img) #Genera un histograma de la imagen leida

size = img.size #tamano de la imagen en pixeles
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

#Mapea la imagen equalizada
imgEqualizedHist = img.copy()
for i in range(len(imgEqualizedHist)):
    for j in range(len(imgEqualizedHist[0])):
        imgEqualizedHist[i][j] = Sk[img[i][j]]


#Histogramas generados y equalizados
histEqualized = generateHistogram(imgEqualizedHist)
histOriginal = returnHistogram(hist)
histFinal = returnHistogram(histEqualized)


#Concatena las imagenes en una sola ventana
row1 = np.concatenate((cv.resize(img,(300,300),interpolation = cv.INTER_CUBIC), np.zeros((300,5), dtype='uint8'), cv.resize(imgEqualizedHist,(300,300),interpolation = cv.INTER_CUBIC)), axis = 1)
row2 = np.concatenate((cv.resize(histOriginal,(300,300),interpolation = cv.INTER_CUBIC), np.zeros((300,5), dtype='uint8'), cv.resize(histFinal,(300,300),interpolation = cv.INTER_CUBIC)), axis = 1)
finalImg = np.concatenate((row1,row2), axis=0)
frame = cv.vconcat([row1,row2])

cv.imwrite("results/a.jpg", frame) #Guarda el resultado como imagen
cv.imshow("Image Histogram", finalImg) #Muestra la imagen resultado
cv.waitKey(0)
cv.destroyAllWindows()
