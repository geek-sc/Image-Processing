#------------------------------------------
#Librerias
import numpy as np
import random
import cv2 as cv
import math
#------------------------------------------
#Binariza la imagen
def umbralize(img,umbral):
	imgResult=img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img [i,j] >= umbral :
				imgResult[i,j] = 255
			else :
				imgResult[i,j] = 0
	return imgResult
#-------------------------------------------------
#Creacion de kernel
def create_kernel(forma,size):
	if forma == 'cross':
		kernel = np.zeros((size,size),np.uint8)
		for i in range(size):
			if size//2 ==i:
				for j in range(size):
					kernel[i,j] = 255
			else:
				kernel[i,size//2]= 255
	if forma == 'square':
		kernel = np.ones((size,size), np.uint8)*255
	return kernel
#--------------------------------------------------
def padding(img,fil,col):
	padding_img = np.zeros((img.shape[0]+(fil-1),img.shape[1]+(col-1)),np.uint8)
	padding_img[(fil-1)//2:padding_img.shape[0]-(fil-1)//2,(col-1)//2:padding_img.shape[1]-(col-1)//2]=img
	return padding_img

#---------------------------------------------------------------------
def compare_erosion(win, kernel):
	for i in range(win.shape[0]):
		for j in range(win.shape[1]):
			if kernel[i,j] != 0:
				if kernel[i,j] != win[i,j]:
					return False
	return True
#---------------------------------------------------------------------
def erosion(img,kernel,iterations):
	filas =kernel.shape[0]
	cols  =kernel.shape[1]
	for l in range(iterations):
		paddedimg = padding(img,filas,cols)
		newIMG    = np.zeros(img.shape,np.uint8)
		for i in range((filas-1)//2,paddedimg.shape[0]-(filas-1)//2):
			for j in range((cols-1)//2,paddedimg.shape[1]-(cols-1)//2):
				ventana=paddedimg[i-((filas-1)//2):i+((filas-1)//2)+1,j-(cols-1)//2:j+(cols-1)//2+1]
				if compare_erosion(ventana,kernel):
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 255
				else:
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 0
		#img=newIMG
	return newIMG
#------------------------------------------------------------------
#Comparacion logica AND para Dilatacion
#Obtiene todos los valores verdaderos comparando 2 arrays de igual tamaño
def comparison_dilation(win, kernel):
	for i in range(win.shape[0]):
		for j in range(win.shape[1]):
			if kernel[i,j] != 0:
				if kernel[i,j] == win[i,j]:
					return True
	return False

#---------------------------------------------------------------------
#Funcion de Dilatacion
def dilation(img,kernel,iterations):
	filas =kernel.shape[0]
	cols  =kernel.shape[1]
	for l in range(iterations):
		paddedimg = padding(img,filas,cols)
		newIMG    = np.zeros(img.shape,np.uint8)
		for i in range((filas-1)//2,paddedimg.shape[0]-(filas-1)//2):
			for j in range((cols-1)//2,paddedimg.shape[1]-(cols-1)//2):
				ventana=paddedimg[i-((filas-1)//2):i+((filas-1)//2)+1,j-(cols-1)//2:j+(cols-1)//2+1]

				#ord = comparison(ventana,kernel)
				#if True in ord:
				if comparison_dilation(ventana,kernel):
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 255
				else:
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 0
		#img=newIMG
	return newIMG

#----------------------------------------------------
#Closing: dilation - erosion
def closing(img):
    img = img.copy()
    result = np.zeros(img.shape)
    kernel = create_kernel('square',5) #kernel cuadrado
    dil = dilation(img, kernel,5)
    im2 = dil.copy()
    erode = erosion(im2, kernel,7)
    result = erode.copy()

    return result

#-------------------------------------------------------
def opening(img):
    img = img.copy()
    result = np.zeros(img.shape)
    kernel = create_kernel('square',5) #kernel cuadrado
    erode = erosion(img, kernel,2)
    dil = erode.copy()
    dil = dilation(erode,kernel,2)
    result = dil.copy()

    return result

#------------------------------------------------------------

def detect_circles (img, rgb, seg):

	img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


	#Transformada de Hough para detectar circulos
	detected_circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 15, param1 = 130,param2 = 12, minRadius = 3, maxRadius = 12)


	#variables para determinar circulo mas grande y circulo mas pequeno
	r_menor=10
	r_mayor=0
	B_S_circle=[[0,0,0],[0,0,0]]

	#------------------------------
	# parametros para cv.putText
	#------------------------------
	# font
	font = cv.FONT_HERSHEY_SIMPLEX
	# fontScale
	fontScale = 1
	# Blue color in BGR
	color = (0, 255, 0)
	# Line thickness of 2 px
	thickness = 2
	#-------------------------------

	if detected_circles is not None:

		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
			if r < r_menor: #condicion para detectar el mas pequeno
				r_menor=r
				B_S_circle[0][0]=pt[0]
				B_S_circle[0][1]=pt[1]
				B_S_circle[0][2]=pt[2]
			if r >= r_mayor: #condicion para detectar el mas grande
				r_mayor = r
				B_S_circle[1][0]=pt[0]
				B_S_circle[1][1]=pt[1]
				B_S_circle[1][2]=pt[2]


		#almacena los circulos detectados
		detected_circles = np.uint16(np.around(detected_circles))

		cont_circles=0 #contador para circulos detectados
		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
			#Dibuja los circulos
			c = cv.circle(rgb, (a, b), r, (235, 235, 50), 2)
			cont_circles += 1

		#guarda como imagen los circulos detectados
		cv.imwrite('circles.png',rgb)

		#Color azul para el circulo mas pequeno
		for i in range (1,int(B_S_circle[0][2])+1):
			cv.circle(rgb, (int(B_S_circle[0][0]), int(B_S_circle[0][1])), i, (0, 0, 255), 2)

		#Color rojo para el circulo mas grande
		for i in range (1,int(B_S_circle[1][2])+2):
			cv.circle(rgb, (int(B_S_circle[1][0]), int(B_S_circle[1][1])), i, (255, 0, 0), 2)
			#Insertar texto con cv2.putText()
			cv.putText(rgb, 'B',(B_S_circle[1][0],B_S_circle[1][1]), font,fontScale, color, thickness, cv.LINE_AA)


	#Imprime el numero de granos d epolen detectados
	#text = "THERE ARE",cont_circles, "GRAINS POLLEN "
	#print("THERE ARE",cont_circles, "GRAINS POLLEN ")
	text = '               Pollen grains detected: '+ str(cont_circles)
	cv.putText(seg,text, (20,30), font,0.7, (255, 255, 255), 1, cv.LINE_AA)
	cv.imwrite('text.jpg',seg)
	return(rgb)

#-----------------------------------------------
def log_transf(img, c):
	rows,cols=img.shape
	#Initialize variables
	imgResult=np.zeros((rows,cols),dtype="uint8")

	#Logaritmic image
	# S = c * log (1 + f(x,y))
	for i in range(rows):
	    for j in range(cols):
	        imgResult[i,j] = c*np.log10(1+img[i,j])

	return imgResult

#----------------------------------------------

#===============================================================
#                   MAIN
#====================================================================
rgb = cv.imread('pollengrains.jpeg',1)
ori = cv.imread('pollengrains.jpeg',1)
img = cv.imread('pol.jpg',0)
text = cv.imread('text.jpg',1)

log = log_transf(img, 188)
#cv.imshow('log',log)
line = np.zeros((50,750,3), np.uint8)


img_blur = cv.GaussianBlur(log, (7, 7), 0)
th_img = umbralize(img_blur, 180) # 160 Segmenta la imagen: polen color negro

#ero = erosion(th_img, create_kernel('cross',3))
closed = closing(th_img) #closing con kernel (5,5) cuadrado
ero = dilation(closed, create_kernel('cross',3),2)

mask = ero.copy()
f = cv.add(mask,log)
#cv.imshow('mask', f)
cv.imwrite('f.png',f)
final = cv.imread('f.png', 1)

fin = detect_circles(final,rgb,line)
#cv.imshow('fin',fin)

circles = cv.imread('circles.png', 1)

row1 = cv.hconcat([img, log, img_blur])
row2 = cv.hconcat([th_img,closed,f])
r = cv.vconcat([row1,row2])
cv.imshow('result',r)

r3 = cv.hconcat([ori, circles, fin])
r4 = cv.vconcat([r3,text])
cv.imshow('detection',r4)


cv.waitKey(0)
cv.destroyAllWindows()
