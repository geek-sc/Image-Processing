# -*- coding: utf-8 -*-
"""
K-Means Lab8
"""
import numpy as np
import scipy as scp
import cv2
import sys

global refPt
refPt=[] #almacena las posiciones en las que se hizo click

#========= Funciones para calcular distancias=====================
def euclidean(x,y):
    d=0
    for i in range(3):
        d+=np.power(x[i]-y[i],2.0)
    return np.sqrt(d)

def manhattan(x,y):
    d=0
    for i in range(3):
        d+=np.abs(x[i]-y[i])
    return d
def chebyshev(x,y):
    d=0
    for i in range(3):
        if d<np.abs(x[i]-y[i]):
            d=np.abs(x[i]-y[i])
    return d
def spearman(x,y):
    d=0
    for i in range(3):
        d+=np.power(x[i]-y[i],2.0)
    return d
#======================Funcion K means===========================
"""
Entrada:
* m es la imagen
* c array de colores RGB mediante GUI
* max es el numero maximo de iteraciones
* chose es la opcion del metodo de calculo de distancia

Salida:
* K es un array con las etiquetas segun el numero de c
"""
def Kmeans(m,c,n,max=2,chose=0):
    x,y,z=m.shape
    K=np.ones((x,y), dtype = "uint8") #inicializa K con unos

    count=0
    distancias=np.zeros(n)

    while count<=n:
        count = count+1
        for i in range(x):
            for j in range(y):
                for k in range(n):
                    if chose==0:
                        distancias[k]=euclidean(m[i][j],c[k])
                    elif chose==1:
                        distancias[k]=manhattan(m[i][j],c[k])
                    elif chose==2:
                        distancias[k]=chebyshev(m[i][j],c[k])
                    elif chose==3:
                        distancias[k]=spearman(m[i][j],c[k])
                pos=scp.argmin(distancias)
                K[i][j]=pos
    return K

#=================================================
def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])

def Clickcenters(image):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",click_and_crop)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])
#======================================================
def Center(img): #Calculo de los centroides
    l=len(refPt)
    print('refPt:',refPt)
    print("l:",l)
    c=np.zeros((l,3))
    for i in range(l):
        for j in range(3):
            c[i][j]=img[refPt[i][1]][refPt[i][0]][j]
    return c
#=======================================================
img=cv2.imread('peppers256.png')

Clickcenters(img) #obtiene los puntos
c=Center(img) # calcula los centroides
print('c:',c)
K=Kmeans(img,c,len(refPt)) # calcula K means, con K como etiquetas

#redimensiona a K para poder usado en la nueva imagen
labels = K.reshape((img.shape[:-1]))
reduced = np.uint8(c)[labels] #coloca los colores usando c
cv2.imshow('K-Means', reduced) #Muestra la imagen

cv2.imwrite('K-Means.png',reduced) #resultado con K = 5

cv2.waitKey(0)
cv2.destroyAllWindows()
