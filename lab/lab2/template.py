# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    pass


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    pass


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    


    # Initialize SIFT detector
    


    # Store SIFT keypoints of original image in a Numpy array
    


    # Rotate around point at center of image.
    


    # Degrees with which to rotate image
    


    # Number of times we wish to rotate the image
    

    
    # BFMatcher with default params
    

        # Rotate image
        

        # Compute SIFT features for rotated image
        

        
        # Apply ratio test
        


        # cv2.drawMatchesKnn expects list of lists as matches.
        
