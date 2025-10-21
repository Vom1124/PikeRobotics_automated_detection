"""
Image Morphology Processing for Depth Maps

This script performs morphological operations (erosion and dilation) on depth images
with pre-processing (median blur) to enhance obstacle detection. The operations help:
- Reduce noise in depth measurements
- Enhance significant features
- Improve obstacle boundary definition

Author: Vomsheendhur (Vom)  Raju
Department: Mechanical Engineering
Institution: North Dakota State University, Fargo, ND, USA
Date: June 20, 2024
"""

import numpy as np
import cv2 as cv

# Morphological kernel (3x3 square for erosion/dilation)
kernel_morph = np.ones((3, 3), dtype=np.uint8)

# Alternative sharpening kernel (commented out as unused in final version)
# kernel_sharpen = np.array([[0, -1, 0], 
#                           [-1, 5, -1],
#                           [0, -1, 0]])

def image_morph_algo(img):
    """
    Apply morphological processing pipeline to depth image.
    
    Processing steps:
    1. Median blur (3x3) for noise reduction
    2. Erosion (1 iteration) to remove small artifacts
    3. Dilation (4 iterations) to enhance remaining features
    
    Args:
        img (numpy.ndarray): Input depth image (16UC1 format recommended)
        
    Returns:
        numpy.ndarray: Processed depth image with enhanced features
    """
    # Step 1: Noise reduction with median blur
    img_filtered = cv.medianBlur(img, 5)
    
    # Step 2: Erosion to remove small noise artifacts
    img_eroded = cv.erode(img_filtered, kernel_morph, iterations=7)
    
    # Step 3: Dilation to enhance remaining features
    img_morph = cv.dilate(img_eroded, kernel_morph, iterations=7)
    
    # Optional sharpening (commented out as it may amplify noise)
    # img_morph = cv.filter2D(img_morph, -1, kernel_sharpen)
    
    return img_morph