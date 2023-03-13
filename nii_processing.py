# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import vtk 
from mayavi import mlab


#"C:/Users/grego/Desktop/Gense_Technologies/NII files-20230220T160540Z-001/NII files/achillestendontear_sagittalT1_2.nii"
#"C:/Users/grego/github/EIT_GT_ENPC/achillestendonrupture_sagittal_T2.nii"
nii_file = nib.load("C:/Users/grego/github/EIT_GT_ENPC/NII_GT/healthy_sagittalPD.nii")
image_data = np.array(nii_file.dataobj)
image_mesh = np.array(image_data[:, :, :, 0])
n_row,n_column,nb_slides,_ = image_data.shape

# # Define the depth range to show
# start_depth = 20
# end_depth = 30

# # Slice the image mesh to keep only the desired depth range
# sliced_image_mesh = image_mesh[:, :, start_depth:end_depth]


# Create the 3D mesh using mayavi
mlab.contour3d(image_mesh, opacity=0.5)
# Show the resulting 3D mesh
mlab.show()

for k in range(nb_slides):
    test = image_data[:,:,k]
    plt.imshow(test)
    plt.show()
    
    
