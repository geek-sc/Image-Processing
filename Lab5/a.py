import numpy as np
import random
import cv2
import math

#-----------------------------------------------------------
#Funcion para ruido de sal y pimienta
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
#Funcion para generar ruido gaussiano
def gauss_noise(image):
    noisy_image = np.zeros(image.shape,np.uint8)
    row = image.shape[0]
    col = image.shape[1]
    gaussian_noise = np.zeros((row,col),dtype=np.uint8)
    noiseg = cv2.randn(gaussian_noise, 100, 150)
    gaussian = (noiseg*0.5).astype(np.uint8)
    noisy_image = cv2.add(image,gaussian)
    return noisy_image

#-----------------------------------------------------------------
#Funciones: Filtro para imagen con ruido de sal y pimienta

def fill_kernel(kernel_mid_x, kernel_mid_y, kernel_size, image):

    kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    from_x = kernel_mid_x - int(math.floor(kernel_size[1]/2))
    to_x = kernel_mid_x + int(math.floor(kernel_size[1]/2))
    from_y = kernel_mid_y -  int(math.floor(kernel_size[0]/2))
    to_y = kernel_mid_y +  int(math.floor(kernel_size[0]/2))

    for i in range(from_x, to_x + 1):
        for j in range(from_y, to_y + 1):

            kernel[i - from_x, j - from_y] = image[i,j]

    return kernel


def get_median(vector):
    vector = np.sort(vector)
    median = vector[int(math.floor(len(vector)/2))]
    return median


def medianFiltering(filter_size, image):
    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    #asumimos que el kernel es simetrico y dimensiones impares
    edge = int(math.floor(filter_size[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(edge, num_cols - edge):
            kernel = fill_kernel(i,j, filter_size, image)
            filtered_image[i,j] = get_median(kernel.flatten())


    return filtered_image
#-----------------------------------------------------------------
#Funcion: filtro para ruido gaussiano
def harmonicMeanFilter(noise_img, window):
    img = noise_img.copy()
    rows, cols = img.shape

    #the pad refers to the number of external pixels(border) not considered when filter the borders
    pad = int(window/2)


    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            pixels_sum = 0
            for k in range(window):
                for l in range(window):

                    pixels_sum += 1/((img[i-pad+l, j-pad+k])+0.00000001)

            img[i,j] = window*window /(pixels_sum+0.000000001)


    return img
#--------------------------------------------------------------
image = cv2.imread('lena.jpg',0) #imagen leida
sp_img = sp_noise(image,0.10) #imagen con ruido sal y pimienta
gauss_img = gauss_noise(image) #imagen con ruido gaussiano
#cv2.imshow('salt',sp_img)
#----------------------------------------------------------
#Aplicamos filtro para sal y pimienta
kernel_size = (3, 3) #kernel para filtro
saltpepper = sp_img.copy()
median_filtered_image = medianFiltering(kernel_size, saltpepper) #imagen aplicando median filter
#cv2.imshow('median filter',median_filtered_image)
#------------------------------------------------------------
#Aplicamos filtro para imagen ruido gausiano
harmonic_filtered_image = harmonicMeanFilter(gauss_img,3)
#cv2.imshow('harmonic',harmonic_filtered_image)
#-------------------------------------------------------------
#Mostramos las imagenes
row1 = np.hstack((sp_img,gauss_img))
row2 = np.hstack((median_filtered_image,harmonic_filtered_image))
result = np.vstack((row1,row2))
cv2.imshow('Salt and pepper -> Median Filter|| Gaussian -> Harmonic Filter',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
