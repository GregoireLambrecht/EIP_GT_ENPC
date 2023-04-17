import numpy as np 
import cv2

def label_centroid(stats, labels):
    centroids_label = []
    for i in range(0,len(stats)): 
        x, y, width, height, _ = stats[i]
        sub_labels = labels[y:y+height, x:x+width]
        sub_labels = sub_labels[sub_labels != 0]  # remove zero values
        unique_labels, label_counts = np.unique(sub_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]
        centroids_label.append(most_common_label)
        
    return centroids_label
  


CENTERS = []
LABELS = []
NB_CENTERS = []
N_IMG = []
STATS =  []

for k in range(4,14):
    
    img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/raw/achilles tendon rupture/" + str(k) +".jpeg", cv2.IMREAD_GRAYSCALE) 
    #put a filter
    img = threshold(img,100)

    #segmentation
    n_img,labels, stats, centroids = segmentation(img, show = False)
    
    labels = label_centroid(stats[1:], n_img[1:])
    N_IMG.append(n_img)
    CENTERS.append(centroids[1:])
    LABELS.append(labels)
    NB_CENTERS.append(len(centroids[1:]))
    STATS.append(stats)
    
    
indice_train = 0

img_train = N_IMG[indice_train]

cv2.imshow('Image-Reference', img_train)
cv2.waitKey(0)
cv2.destroyAllWindows()


centroids_train = np.array(CENTERS[indice_train], dtype=np.float32)

labels_train = np.array(LABELS[indice_train], dtype=np.float32)

# Create a K-NN model
model = cv2.ml.KNearest_create()

# Train the model on the train data
model.train(centroids_train, cv2.ml.ROW_SAMPLE, labels_train)

k=1

DIST = []
RESULTS = []
NEIGHBOURS = []

for i in range(0,len(CENTERS)): 
    
    # Convert input data to correct data type and reshape
    test_samples = np.array(CENTERS[i], dtype=np.float32)
    
    
    # Find the k nearest neighbors of each point in SECOND
    # results = The label given to the new-comer depending upon the kNN theory we saw earlier. If you want the Nearest Neighbour algorithm, just specify k=1.
    # neighbours = The labels of the k-Nearest Neighbours.
    # dist = The corresponding distances from the new-comer to each nearest neighbour.
    ret, results, neighbours ,dist= model.findNearest(test_samples, k)
    
    DIST.append(dist)
    RESULTS.append(results)
    NEIGHBOURS.append(neighbours)
    
    indice_train = i
    
    centroids_train = np.array(CENTERS[indice_train], dtype=np.float32)

    labels_train = np.array(neighbours, dtype=np.float32)

    # Create a K-NN model
    model = cv2.ml.KNearest_create()

    # Train the model on the train data
    model.train(centroids_train, cv2.ml.ROW_SAMPLE, labels_train)
    
    
    

IMAGE_SEG = []
for j in range(len(CENTERS)): 
    if j == indice_train:
        continue
    
    image_1 = N_IMG[j]

    centroids_1 = CENTERS[j]
    labels_1 = LABELS[j]
    
    for i in range(len(centroids_1)): 
        image_1[image_1 == labels_1[i]] = NEIGHBOURS[j][i]
    
    IMAGE_SEG.append(image_1)
    cv2.imshow('Result for image '+ str(j), image_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("C:/Users/grego/github/EIT_GT_ENPC/achilles_tendon_rupture_TIFF/"+str(i+4) + ".tif", image_1)
    
    print(len(np.unique(image_1)))







