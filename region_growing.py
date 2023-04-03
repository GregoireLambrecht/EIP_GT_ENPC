import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6_filtre.jpeg", 0)

plt.imshow(img)

# Set the starting seed point for region growing
seed_point = (450, 400)

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
        
        
#erosion method
def erode(img, show = False):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def put_in_white_black(img):
    img_white = img.copy()
    for x in range(len(img)):
        for y in range(len(img)):
            if img[x,y]<127 :
                img_white[x,y] = 0
            else :
                img_white[x,y] = 255
    return img_white
                
                

def segmentation(img,n):
    img = erode(img)  # add erosion step
    put_in_white_black(img)
    size = len(img)
    dx = int(size/n)
    fill_color = 220
    for x in range(n):
        for y in range(n):
            seed_point = (x*dx,y*dx)
            if img[x*dx,y*dx] == 255 : 
                # print(seed_point)
                # print(img[x*dx,y*dx])
                #Perform region growing
                find_surface(img, seed_point,fill_color)
                fill_color = fill_color-1
    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                
# img = put_in_white_black(cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6_filtre.jpeg", 0))
# find_surface(img, seed_point,show = True)

img = put_in_white_black(cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6_filtre.jpeg", 0))
segmentation(img,10)

#img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6_filtre.jpeg", 0)


        
                

            