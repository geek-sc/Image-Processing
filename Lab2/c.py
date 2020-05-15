import numpy as np
import cv2


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



img = cv2.imread('imgs/lena.jpg', 0)


horizontal_gradient = gradient_x(img)
vertical_gradient = gradient_y(img)
gradient = horizontal_gradient + vertical_gradient
average_filtered = average_filter(img, 3)
sobel_filtered = sobel_filter(average_filtered)

frame = cv2.hconcat([img, horizontal_gradient, vertical_gradient, gradient, sobel_filtered])
cv2.imwrite("results/c.jpg", frame)
cv2.imshow("Spatial Filters", frame)
cv2.waitKey(0)
