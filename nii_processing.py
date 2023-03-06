# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import vtk 

nii_file = nib.load("C:/Users/grego/github/EIT_GT_ENPC/achillestendonrupture_sagittal_T2.nii")
image_data = nii_file.get_fdata()
n_row,n_column,nb_slides,_ = image_data.shape



# convert the numpy array to float
image_data = image_data.astype(np.float32)

# create a vtkImageData object from the numpy array
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(n_row, n_column, nb_slides)
vtk_image.SetSpacing(nii_file.header.get_zooms()[:3])
vtk_array = vtk.vtkFloatArray()
vtk_array.SetNumberOfComponents(1)
vtk_array.SetArray(image_data.ravel(), image_data.size, 1)
vtk_image.GetPointData().SetScalars(vtk_array)

# create the marching cubes filter and set the parameters
marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputData(vtk_image)
marching_cubes.ComputeNormalsOn()
marching_cubes.SetValue(0, np.max(image_data) / 2.0)

# create a vtkPolyData object from the output of the marching cubes filter
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(marching_cubes.GetOutput().GetPoints())
poly_data.SetPolys(marching_cubes.GetOutput().GetPolys())


# create a vtkRenderer and a vtkRenderWindow
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# create a vtkActor from the vtkPolyData object and add it to the renderer
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer.AddActor(actor)

# set up the camera position and render the window
camera = renderer.GetActiveCamera()
camera.SetPosition(image_data.shape[0], image_data.shape[1], image_data.shape[2])
camera.SetFocalPoint(0, 0, 0)
renderer.ResetCamera()
render_window.Render()

# create a vtkInteractorStyleTrackballCamera object
style = vtk.vtkInteractorStyleTrackballCamera()


# start the interactive window
render_window_interactor = vtk.vtkRenderWindowInteractor()


# create a vtkInteractorStyleTrackballCamera object
style = vtk.vtkInteractorStyleTrackballCamera()

# set the interactor style and start the interactor
render_window_interactor.SetInteractorStyle(style)
render_window_interactor.Start()


# for k in range(nb_slides):
#     test = image_data[:,:,k]
#     plt.imshow(test)
#     plt.show()
    
    
# i0 = image_data[:,:,0]
# plt.imshow(i0)
# plt.show()
