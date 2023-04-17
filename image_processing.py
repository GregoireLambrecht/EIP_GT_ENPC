import cv2
import numpy as np

##############################################################################
#RUN BEFORE RUNING OTHERS FILES
##############################################################################


#Put an image in black and white
def put_in_white_black(img):
    img_white = img.copy()
    for x in range(len(img)):
        for y in range(len(img)):
            if img[x,y]<127 :
                img_white[x,y] = 0
            else :
                img_white[x,y] = 255
    return img_white
               

# Apply thresholding to create a binary image
#better than put_in_white_black
def threshold(img,seuil):
    thresh_value, binary_image = cv2.threshold(img, seuil, 255, cv2.THRESH_BINARY)
    return binary_image

#erosion method
def erode(img, kern = 7, iteration = 1, show = False):
    kernel = np.ones((kern,kern), np.uint8)
    img = cv2.erode(img, kernel, iterations=iteration)
    return img


#dilation method
def dilation(img, kern = 6, iteration = 2, show = False):
    kernel = np.ones((kern,kern), np.uint8)
    img = cv2.dilate(img, kernel, iterations= iteration)
    if show: 
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    return img
    


def verifBW(img): 
    for x in range(len(img)):
        for y in range(len(img)):
            if img[x,y]!=0 and img[x,y]!=255:
                print(x,y)
                print(img[x,y])
    