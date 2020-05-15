import numpy as np
import cv2
import math



def exp_transform(img, exp):
    rows, cols = img.shape
    imgResult=np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            imgResult[i,j]= pow(img[i,j], exp)
            if imgResult[i,j] >= 255:
                imgResult[i,j]=255
    return imgResult

def spatial_convolution(img, kernel, padding="same"):
    result_image = np.zeros_like(img)
    if padding == "same":
        offset = kernel.shape[0]//2
        padded_image = np.zeros((img.shape[0]+2*offset, img.shape[1]+2*offset), dtype="uint8")
        padded_image[offset:img.shape[0]+offset, offset:img.shape[1]+offset] = img
        for row in range(1, img.shape[0] ):
            for col in range(1, img.shape[1] ):
                value = kernel * padded_image[(row - 1):(row + 2), (col - 1):(col + 2)]
                result_image[row, col] = min(255, max(0, value.sum ()))
    return result_image


def average_filter(img, kernel_size, padding="same"):
    assert img.shape[0] == img.shape[1], "Image must be of equal size"
    kernel = np.ones((kernel_size,kernel_size)) * (1/(kernel_size*kernel_size))
    return spatial_convolution(img, kernel, padding=padding)

def median_filter(img, kernel_size, padding='same'):
    assert img.shape[0] == img.shape[1], "Image must be of equal size"
    kernel = np.zeros((kernel_size,kernel_size)) * (1/(kernel_size*kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 1
    return spatial_convolution(img, kernel, padding=padding)

def laplacian_filter(img, padding='same'):
    assert img.shape[0] == img.shape[1], "Image must be of equal size"
    kernel = np.zeros((3, 3))
    kernel[1, 1] = -4
    kernel[1, 0] = 1
    kernel[1, 2] = 1
    kernel[0, 1] = 1
    kernel[2, 1] = 1
    return spatial_convolution(img, kernel, padding=padding)

def laplacian_enhancement(img, padding='same'):
    assert img.shape[0] == img.shape[1], "Image must be of equal size"
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 5
    kernel[1, 0] = -1
    kernel[1, 2] = -1
    kernel[0, 1] = -1
    kernel[2, 1] = -1
    return spatial_convolution(img, kernel, padding=padding)

def gradient_y(img, padding='same'):
    kernel = [[-1,-2,-1], [0,0,0], [1,2,1]]
    kernel = np.array(kernel)
    return spatial_convolution(img, kernel, padding=padding)

def gradient_x(img, padding='same'):
    kernel = [[-1,0,1], [-2,0,2], [-1,0,1]]
    kernel = np.array(kernel)
    return spatial_convolution(img, kernel, padding=padding)

def sobel_filter(img, padding='same'):
    assert img.shape[0] == img.shape[1], "Image must be of equal size"
    return abs(gradient_x(img, padding=padding)) + abs(gradient_y(img, padding=padding))



img = cv2.imread("Lab3/coin1A.jpg", 0)
img1 = laplacian_enhancement(img)
img2 = average_filter(img1, 3)
img3 = exp_transform(img2, 1.1)
img4 = img3*3
img5 = average_filter(img4, 3)


row1 = np.concatenate((cv2.resize(img,(200,200),interpolation = cv2.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv2.resize(img1,(200,200),interpolation = cv2.INTER_CUBIC)), axis = 1)
row2 = np.concatenate((cv2.resize(img2,(200,200),interpolation = cv2.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv2.resize(img3,(200,200),interpolation = cv2.INTER_CUBIC)), axis = 1)
row3 = np.concatenate((cv2.resize(img4,(200,200),interpolation = cv2.INTER_CUBIC), np.zeros((200,10), dtype='uint8'), cv2.resize(img5,(200,200),interpolation = cv2.INTER_CUBIC)), axis = 1)

finalImg = np.concatenate((row1,row2,row3), axis=0)
#finalImg = np.concatenate((row1,row2,row3), axis=0)
cv2.imshow("spatial enhancement methods", finalImg)
cv2.waitKey(0)
