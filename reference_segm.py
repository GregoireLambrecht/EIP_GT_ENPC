import cv2


####################################################################
#Knn on the reference image 
####################################################################


def labelize(image_reference, centroids_reference, labels_reference, img, show = True, name = ""):
    #put a filter
    img = threshold(img,90)
       
    #segmentation
    n_img,labels, stats, centroids = segmentation(img, show = False,er_kern = 10, di_kern = 5)
    
    centroids = centroids[1:]
    stats = stats[1:]
    
    labels = label_centroid(stats, n_img,show = False)                              #At one, to avoid zero (the black part)
    
    #training set
    centroids_train = np.array(centroids_reference, dtype=np.float32)
    
    #training label
    labels_train = np.array(labels_reference, dtype=np.float32)
    
    # Create a K-NN model
    model = cv2.ml.KNearest_create()
    
    # Train the model on the train data
    model.train(centroids_train, cv2.ml.ROW_SAMPLE, labels_train)
    k=1
    
    ret, results, neighbours ,dist= model.findNearest(np.array(centroids, dtype=np.float32), k)
    
    
    for i in range(len(labels)):
          n_img[n_img == labels[i]] = results[i]
         
    if show: 
        cv2.imshow('Result for image ' + name, n_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return n_img, labels, centroids


for k in range(0,16):
    img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/raw/achilles tendon rupture/" + str(k) +".jpeg", cv2.IMREAD_GRAYSCALE) 
    n_img,_,_ = labelize(image_reference, centroids_reference, labels_reference, img)
    
    cv2.imwrite("C:/Users/grego/github/EIT_GT_ENPC/achilles_tendon_rupture_reference_TIFF/"+str(k) + ".tif", n_img)



