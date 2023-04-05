import cv2
import numpy as np
import matplotlib.pyplot as plt


#ATTENTION ////////////////////////////////////////////////////////////////////
#YOU HAVE TO RUN image_processing.py before runing this file
#////////////////////////////////////////////////////////////////////////////// 


#PERFORM ONE REGION GROWING 
#img : 2D array containing only 0 and 255. 
#seed_point : origin of the region growing
#show = True : show the result
def find_surface(img,seed_point,fill_color = 100, show = False):
    # Set the threshold for region growing
    threshold = 1
    # Set the connectivity (4 or 8) for region growing
    connectivity = 4
    # Initialize the mask for region growing
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    # Perform region growing only on white regions
    if img[seed_point[1], seed_point[0]] == 255:
        cv2.floodFill(img, mask, seed_point, fill_color, threshold, threshold, connectivity)
    if show :
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


#PERFORM REGION GROWING 
#img : 2D array containing only 0 and 255
def segmentationbis(img): 
    img = erode(img,kern = 7,  iteration=1)                                    # add erosion step
    img = dilation(img, kern = 3, iteration = 1)                               # add dilation step
    # Perform region growing
    connectivity = 4                                                          #4 or 8 : impact the definition of two nearby pixel                     
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    fill_color = 250
    #COLOR THE IMAGE 
    for X in centroids: 
        img = find_surface(img, (int(X[0]),int(X[1])),fill_color)
        fill_color -= 5  
    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img,labels,centroids
        
###############################################################################
###############################################################################              
# Load image as grayscale
img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6.jpeg", cv2.IMREAD_GRAYSCALE) 
#put a filter
img = threshold(img,100)

plt.imshow(img)

#segmentation
n_img,labels, centroids = segmentationbis(img)



        
                

            