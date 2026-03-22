# Canny Edge Detector (Skeleton Code)
# Base code provided for this project

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 

def load_and_display_image(image_path):
    img = 0
    return img

def gaussianFilter(img, kernelSize, stdDev):
    gImg = img 

    cv2.imwrite("blurred_Image.png", gImg)
    plt.imshow(gImg, cmap='gray')
    plt.title("Blurred Image")
    plt.show()

    return gImg

def applyKernel(img, kernel):
    modImg = img
                    
    return modImg 

def sobel(img):
    
    sxImg  = img
    cv2.imwrite("sobelX_Image.png", sxImg)


    syImg  = img
    cv2.imwrite("sobelY_Image.png", syImg)


    magArr = img
    dirArr = img;


    plt.subplot(1, 3, 1)
    plt.imshow(sxImg, cmap='gray')
    plt.title("Sobel X Image")

    plt.subplot(1, 3, 2)
    plt.imshow(syImg, cmap='gray')
    plt.title("Sobel Y Image")

    plt.subplot(1, 3, 3)
    plt.imshow(magArr, cmap='gray')
    plt.title("Gradient Insensity Image")

    plt.show()

    return(magArr, dirArr)

def nonMaxSupress(magArr, dirArr):
    thinnedImg = magArr
    
    cv2.imwrite("thinned_Image.png", thinnedImg)
    plt.imshow(thinnedImg, cmap='gray')
    plt.title("Thinned Image")
    plt.show()  

    return(thinnedImg)

def doubleThreshold(img, minThreshold, maxThreshold):
    cannyImg = img

    cv2.imwrite("thinned_Image.png", cannyImg)
    plt.imshow(cannyImg, cmap='gray')
    plt.title("Final Image")
    plt.show()  

    return(cannyImg)

if __name__ == "__main__":
    image_path = "your_path_here"
    greyImg = load_and_display_image(image_path)
    blurredImg = gaussianFilter(greyImg, 0, 0)
    mgArr, drArr = sobel(blurredImg)
    thinnedImg = nonMaxSupress(mgArr, drArr)
    finalImg = doubleThreshold(thinnedImg, 0, 0)
