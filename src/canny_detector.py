# Canny Edge Detector - Intro to Robotics Project
# Apply Kernels and various filters to extract edges and lines from photos
# No OpenCV Required!
# Author: Joey P
# Date: 4/15/25 (Updated 3/22/26)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 
import os

# directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_display_image(image_path):
    # import image and grayscale
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
    return img

def applyKernel(img, kernel):
    kernel = np.flipud(np.fliplr(kernel))

    # dimensions
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    blank = np.zeros_like(img, dtype=np.float32)

    # add 0 border pixels
    padded_img = np.pad(img,
                        ((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+kernel_h, j:j+kernel_w]
            blank[i, j] = np.sum(region * kernel)

    modImg = blank
                    
    return modImg

def gaussianFilter(img, kernelSize, stdDev, name="default"):
    # Gaussian kernel
    ax = np.linspace(-(kernelSize // 2), kernelSize // 2, kernelSize)
    xx, yy = np.meshgrid(ax, ax)
    gkernel = np.exp(-(xx**2 + yy**2) / (2. * stdDev**2))

    gImg = applyKernel(img, gkernel)

    print("Gauss kernel applied to {name}")

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"blurred_{name}.png"), gImg)
    plt.imshow(gImg, cmap='gray')
    plt.title("Blurred Image")
    plt.show()

    plt.close()

    return gImg

def sobel(img, name="default"):
    # SobelX Kernel
    sobelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # SobelY Kernel
    sobelY = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # apply Gx and Gy
    Gx = applyKernel(img, sobelX)
    sxImg = Gx
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sobelX_{name}.png"), sxImg)

    Gy = applyKernel(img, sobelY)
    syImg  = Gy
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sobelY_{name}.png"), syImg)

    # magnitude and direction
    magArr = np.sqrt(Gx**2 + Gy**2)
    dirArr = np.arctan2(Gy, Gx)         # radians

    print("Gx and Gy applied to {name}")
    
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

    plt.close()

    return(magArr, dirArr)

def nonMaxSupress(magArr, dirArr, name="default"):
    h, w = magArr.shape
    output = np.zeros((h,w), dtype=np.float32)      # blank array to apply nonMaxSupress
    angle = np.degrees(dirArr) % 180                # radians to degrees

    q = None                                        # left pixel
    r = None                                        # right pixel

    # see if neighboring pixels direction are similar to current direction
    for i in range(1, h-1):
        for j in range(1, w-1):

            # suppressed pixel values
            q = 255                     
            r = 255                     

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = magArr[i, j + 1]
                r = magArr[i, j - 1]

            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = magArr[i + 1, j - 1]
                r = magArr[i - 1, j + 1]

            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = magArr[i + 1, j]
                r = magArr[i - 1, j]

            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = magArr[i - 1, j - 1]
                r = magArr[i + 1, j + 1]

            # suppress pixel if neighboring pixels are larger magnitude
            if magArr[i, j] >= q and magArr[i, j] >= r:
                output[i, j] = magArr[i, j]
            else:
                output[i, j] = 0


    thinnedImg = output

    print("NMS applied to {name}")
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"thinned_{name}.png"), thinnedImg)
    plt.imshow(thinnedImg, cmap='gray')
    plt.title("Thinned Image")
    plt.show()  

    plt.close()

    return(thinnedImg)

def doubleThreshold(img, minRatio, maxRatio, name="default"):

    # values below minVal are suppressed, above maxVal are strong line
    # in between are weak lines
    highThresh = img.max() * maxRatio               # find maxVal
    lowThresh = highThresh * minRatio               # find minVal
    strong = 255                                    # make a strong line
    weak = 75                                       # make a weak line (only kept if connected to strong)

    edges = np.zeros_like(img, dtype=np.uint8)     # result same size as img

    # find strong pixels above threshold and weak pixels between thresholds (hysteresis)
    strong_i, strong_j = np.where(img >= highThresh)
    weak_i, weak_j = np.where((img <= highThresh) & (img >= lowThresh))
    
    edges[strong_i, strong_j] = strong              # strong edge pixels to 255
    edges[weak_i, weak_j] = weak                    # weak edge pixels to 75

    h, w = img.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if edges[i, j] == weak:
                # check 8-connected pixels for strong edge
                if ((edges[i + 1, j -1] == strong) or 
                    (edges[i + 1, j] == strong) or 
                    (edges[i + 1, j + 1] == strong) or 
                    (edges[i, j - 1] == strong) or 
                    (edges[i, j + 1] == strong) or 
                    (edges[i - 1, j - 1] == strong) or 
                    (edges[i - 1, j] == strong) or 
                    (edges[i - 1, j + 1] == strong)):
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0

    cannyImg = edges

    print("Double thresh + Hysteresis complete for {name}")

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"final_canny_{name}.png"), cannyImg)
    plt.imshow(cannyImg, cmap='gray')
    plt.title("Final Image")
    plt.show()  

    plt.close()

    return(cannyImg)

def run_canny_pipeline(filename, k_size, std, low, high, resize_dim=None):
    in_path = os.path.join(INPUT_DIR, filename)
    img_name = os.path.splitext(filename)[0]

    print(f"Processing: {filename}...")
    
    grey = load_and_display_image(in_path)
    if grey is None: return
    
    if resize_dim:
        grey = cv2.resize(grey, resize_dim)
        
    blurred = gaussianFilter(grey, k_size, std, name=img_name)
    mag, dr = sobel(blurred, name=img_name)

    thinned = nonMaxSupress(mag, dr, name=img_name)
    final = doubleThreshold(thinned, low, high, name=img_name)
    
    # Save to output folder
    out_name = f"canny_{filename}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), final)
    print(f"Success! Saved to {OUTPUT_DIR}/{out_name}")

if __name__ == "__main__":
    # 1. Sanity Check
    run_canny_pipeline("sanityCheck.png", 3, 0.5, 0.05, 0.15)

    # 2. Low Contrast
    run_canny_pipeline("lowConstrastLenna.png", 3, 1.25, 0.4, 0.15)
    
    # 3. Noisy Image
    run_canny_pipeline("noisyCameraman.png", 3, 1.5, 0.05, 0.5)
    
    # 4. Lift Bridge (High Shadows)
    run_canny_pipeline("highShadowsLiftBridge.png", 5, 1.5, 0.25, 0.5, resize_dim=(600,400))
    
    # 5. Choice Image
    run_canny_pipeline("IMG_3964.jpg", 3, 1, 0.6, 0.25, resize_dim=(500,900))
