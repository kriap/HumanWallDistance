import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

def reduce_background(img, lower_bound = 1, upper_bound=4):
    
    """
    Function that reduces background noise in the depth image
    """
    
    assert(len(np.shape(img)) == 2), "Accepts only 2 channel depth image"
    h, w = np.shape(img)
    
    for i in range(h):
        
        for j in range(w):
            
            if img[i][j] > upper_bound :
                
                img[i][j] = 255
                
            elif img[i][j] < lower_bound:
                
                img[i][j] = 0
                
            else:
                
                img[i][j] =127
    
    return img

def get_clearence(roi):
    
    #Get Counters using OpenCV
    img2, contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Sort the counters in decreasing order of area
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

    screenCnt = None
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
    
        epsilon = 0.01*cv2.arcLength(c,True)
        contours_poly[i] = cv2.approxPolyDP(c, epsilon, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    #Find the distance between human and shelf
    left = boundRect[0][0]-60

    #Find the distance between human and wall
    right = 120-(boundRect[0][0]+boundRect[0][2])

    if left > right :
    
        print("left",left*1.5/60) 
    
    else:
    
        print("right",right*1.5/60)


def main(argv):
    
    file = argv[1]
    img = np.loadtxt(file)

    img1 = reduce_background(img)
    img1 = np.uint8(img1)

    #Find mean value in the image
    avg = np.mean(img1)

    #Get Canny Edges thresholded by mean +- 25%
    img1 = cv2.Canny(img1, avg - 0.25*avg , avg + 0.25*avg)

    # Get Dimensions of the Image
    h, w = np.shape(img1)

    #Get Region of interest where human can be present from depth image
    roi = np.zeros(shape=(h,w),dtype=np.uint8)
    roi[30:108,60:120]=img1[30:108,60:120]

    get_clearence(roi)
    

if __name__ == "__main__":
    main(sys.argv)
    
    