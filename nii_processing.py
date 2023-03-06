# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import vtk 




nii_file = nib.load("C:/Users/grego/github/EIT_GT_ENPC/achillestendonrupture_sagittal_T2.nii")
image_data = np.array(nii_file.dataobj)
n_row,n_column,nb_slides,_ = image_data.shape



for k in range(nb_slides):
    test = image_data[:,:,k]
    plt.imshow(test)
    plt.show()
    
    
