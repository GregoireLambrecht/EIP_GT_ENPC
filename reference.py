import numpy as np
import cv2

####################################################################
#CREATE AN ARBITRARY REFERENCE IMAGE
####################################################################
def putSquare(top_left_corner,right_down_corner, gap = 3, size = 10):
    lx,ly = top_left_corner
    rx,ry = right_down_corner
    PART = []
    for x in range(lx,rx-size,size+gap):
        for y in range(ly, ry-size, size+gap): 
            PART.append([(xx, yy) for xx in range(x,x+size) for yy in range(y,y+size)])
    return PART
        
        
# Create an image with a black background
#do not forget to precise dtype = np.uint8. 
#Others types don't allow grey scale
image_reference = np.zeros(shape=(630,630), dtype=np.uint8)

# Define the parts of the foot and their corresponding colors
part_colors = {"b": 50, "t": 250, "m": 150}


TENDONS = putSquare((70,490),(470,570))
BONES = putSquare((70,410),(380,490)) 
MUSCLES = putSquare((270,120),(420,400)) + putSquare((420,70),(470,430))


# Define the coordinates for each part of the foot
parts = {"t":TENDONS, "b": BONES, "m": MUSCLES}
    
# Draw each part of the foot with its corresponding color
for part, list_coords in parts.items():
    color = part_colors[part]
    for coords in list_coords:
        for x, y in coords:
            image_reference[x, y] = color

#Segmentation on our reference image
n_img,labels_reference, stats_reference, centroids_reference = segmentation(image_reference, show = False, process = False)

#Delete the first centroid which corresponds to the center of the picture
#Correspond to pixels in black 
centroids_reference = centroids_reference[1:]
stats_reference = stats_reference[1:]

#Get label of each centroid
labels_reference = label_centroid(stats_reference, image_reference, show = False)


# Display the image
cv2.imshow('IMAGE REFERENCE', image_reference)
cv2.waitKey(0)
cv2.destroyAllWindows()



def annotate(image, parts):
    image_annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1


    for label, organ in parts.items():
        for sub_organ in organ :
            x,y = sub_organ[len(sub_organ)//2]
            cv2.putText(image_annotated, label, (y, x),
                            font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    return image_annotated

# Annotate the image reference with the labels
image_annotated = annotate(image_reference, parts)
























