"""
Segundo Algoritmo: Componentes Conectados
"""

import cv2 as cv
import numpy as np


#-------------------------------------------------

def umbralize(img,umbral):
	imgResult=img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img [i,j] >= umbral :
				imgResult[i,j] = 255
			else :
				imgResult[i,j] = 0
	return imgResult

#------------------------------------------------------
def create_kernel(name,size):
	if name =="cross":
		kernel = 255*np.ones((size,size),np.uint8)
		for i in range (size):
			if size//2==i:
				for j in range (size):
					kernel[i,j] = 0
			else :
				kernel[i,size//2] = 0
	if name == "square" :
		kernel = 255*np.ones((size,size),np.uint8)
	return kernel

#-----------------------------------------------------

def compare_dilation(window,kernel):
	for i in range(window.shape[0]):
		for j in range(window.shape[1]):
			if kernel[i,j]==0:
				if kernel[i,j] == window[i,j]:
					return True
	return False

def fill_dilation(img,kernel):
	rows = img.shape[0]
	cols = img.shape[1]

	newImg = 255*np.ones(img.shape,np.uint8)
	for row in range(rows):
		for col in range(cols-1):
			window=img[row-1:row+2,col-1:col+2]
			if compare_dilation(window,kernel):
				newImg[row,col] = 0
			else:
				newImg[row,col]= 255
	return newImg

#-----------------------------------------------
def makex0(p,img):
	final = 255*np.ones(img.shape,np.uint8)
	final[p[0],p[1]]=0
	return final


def intersecate(x0,Ac):
	x1=255*np.ones(x0.shape,np.uint8)
	for i in range(x0.shape[0]):
		for j in range(x0.shape[1]):
			if x0[i][j] == Ac[i][j] == 0:
				x1[i][j]=0
	return x1

def check_end(img,img1):
	flag=0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] != img1[i][j]:
				flag= 1
	return flag
#--------------------------------------------------
def union (fill,border):
	final = 255*np.ones(fill.shape,np.uint8)
	for i in range(fill.shape[0]):
		for j in range(fill.shape[1]):
			if (fill[i][j] == 0) or (border[i][j] == 0):
				final[i][j]=0
	return final



#-------------------------------------------
# Algoritmo2 : Componentes Conectados
#------------------------------------------
def conn_comp(img, umbral):

	img_u = umbralize(img,umbral)
	img_u_c = 255-img_u

	kernel=create_kernel("square",3)

	#Crea X0
	p =(22,81) #row -> 89
	x0=makex0(p,img_u_c)
	xs=[]
	xs.append(x0)

	#Obtenemos Xn's
	i=0
	end=1
	while (end):#i<65
		dilated=fill_dilation(xs[i],kernel)
		xn=intersecate(dilated,img_u)
		xs.append(xn)
		end=check_end(xs[i],xs[i+1])
		i=i+1

    #Une las imagenes con la original
	final1 = union(xs[i//2],img_u)
	final2 = union(xs[-1],img_u)
	row = np.concatenate((img,img_u,final2),1)

	return row


#----------------------------------------------
#            MAIN
#----------------------------------------------
img2 = cv.imread("huesos.png",0)

umbral = 220
result = conn_comp(img2, umbral)
cv.imshow("2 : Connected components",result)
cv.imwrite('result_b2.jpg',result)
cv.waitKey(0)
cv.destroyAllWindows()
