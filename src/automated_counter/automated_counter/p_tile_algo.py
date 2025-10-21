'''
This function computes the p_tile threshold value for the provided image and the Area of Interest (AOI)
Author: Vom Raju, ME Dept., NDSU, Fargo, ND, USA
'''
import numpy as np

def p_tile_thresh(img, AOI):
  n_pixels = AOI*img.shape[0]*img.shape[1]
  hist = np.histogram(img, bins=range(65535))[0]
  hist = np.cumsum(hist)
  return np.argmin(np.abs(hist-n_pixels))