'''
Actividad 3: Erosion y bordes
'''

#Librerias
import numpy as np
import cv2 as cv

#----------------------------------------------------------------------
def padding(img,fil,col):
	padding_img = np.zeros((img.shape[0]+(fil-1),img.shape[1]+(col-1)),np.uint8)
	padding_img[(fil-1)//2:padding_img.shape[0]-(fil-1)//2,(col-1)//2:padding_img.shape[1]-(col-1)//2]=img
	return padding_img

#---------------------------------------------------------------------
def compare(win, kernel):
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
				if compare(ventana,kernel):
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 255
				else:
					newIMG[i-((filas-1)//2),j-(cols-1)//2] = 0
		#img=newIMG
	return newIMG

#---------------------------------------------------------------------
def create_kernel(forma,size):
	if forma == 'cross':
		kernel = np.zeros((size,size),np.uint8)
		for i in range(size):
			if size//2 ==i:
				for j in range(size):
					kernel[i,j] = 255
			else:
				kernel[i,size//2]=255
	if forma == 'square':
		kernel = np.ones((size,size), np.uint8)*255
	return kernel

#---------------------------------------------------------------------
def umbral(img):
	img2=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j]<=50:
				img2[i][j]=255
			else:
				img2[i][j]=0
	return img2

#----------------------------------------------------------------------
def bordes(segmentada,erosionada):
	fil=segmentada.shape[0]
	col=segmentada.shape[1]
	Rimg=segmentada.copy()
	for i in range(fil):
		for j in range(col):
			seg_val=segmentada[i][j]
			ero_val=erosionada[i][j]
			if seg_val == 255 and ero_val==255:
				Rimg[i][j]=0
	return Rimg

#=========================  MAIN   =============================
image = cv.imread("cameraman.jpg",0)#.astype(np.float32)/255
image = cv.resize(image,(250,250))

image_segmented=umbral(image)

size = 3
erosionImage = erosion(image_segmented,create_kernel('cross',3),size)
Bound_image=bordes(image_segmented,erosionImage)

final=np.concatenate((image,image_segmented,erosionImage,Bound_image),axis=1)
cv.imshow("Result",final)

cv.waitKey(0)
cv.destroyAllWindows()
