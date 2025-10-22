'''
This function computes the p_tile threshold value for the provided image and the Area of Interest (AOI)
Author: Vom Raju, ME Dept., NDSU, Fargo, ND, USA
'''
import numpy as np
import cv2
import os
import sys

# # -----------------------
# Function definition
# # -----------------------
# def p_tile_thresh(img, AOI):
#     n_pixels = AOI * img.shape[0] * img.shape[1]
#     hist = np.histogram(img, bins=range(65535))[0]
#     hist = np.cumsum(hist)
#     return np.argmin(np.abs(hist - n_pixels))

# # -----------------------
# # Load image and apply p-tile threshold
# # -----------------------
# current_directory = os.getcwd()
# template_path = f"{current_directory}/SUNCOR/SUNCOR_24/template_rgb_enhanced.png"

# test = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

# thresh_value = p_tile_thresh(test, 0.6)  # Now this works
# ret, test_seg = cv2.threshold(test, thresh_value, np.max(test), cv2.THRESH_TOZERO)

# cv2.imshow("ptile", test_seg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()