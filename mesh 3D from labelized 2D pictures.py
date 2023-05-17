import numpy as np
from segmentation2D import *
import meshlib.mrmeshpy as mr
import matplotlib.pyplot as plt
import cv2
import meshio
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
from pywavefront import Wavefront
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from PIL import Image
import cc3d

settings = mr.LoadingTiffSettings()

print("start")
IMAGE_SEG = np.asarray(IMAGE_SEG)
extraction_pictures = np.asarray(IMAGE_SEG)
label = np.unique(IMAGE_SEG)
print(len(label))
shape = IMAGE_SEG.shape

for l in label:
    extraction_pictures = np.where(IMAGE_SEG != l, 0, IMAGE_SEG)
    for k in range(shape[0]):
        cv2.imwrite(local_path + "labellised organs/organ"+str(l) + "_number" + str(k) + ".tif", extraction_pictures[k])

image_sequence = []
for k in range(17) :
    img = cv2.imread(local_path + "raw/achilles tendon rupture/" + str(k) + ".jpeg", cv2.IMREAD_GRAYSCALE)
    img = threshold(img, 75)
    img = erode(img, kern=7, iteration=1)
    img = dilation(img, kern=5, iteration=1)
    image_sequence.append(img)
    '''cv2.imwrite(local_path + "3D test/" + str(k) + ".tif", IMG[k])
    cv2.imshow('test' + str(k) ,IMG[k])
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

'''# load images from specified directory
settings.dir = local_path + "3D test"

# specifiy size of 3D image element
settings.voxelSize = mr.Vector3f(1, 1, 15)

#create voxel object from the series of images
volume = mr.loadTiffDir(settings)

#define ISO value to build surface
iso=50.0

#convert voxel object to mesh
mesh=mr.gridToMesh(volume, iso)

#save mesh to .stl file
mr.saveMesh(mesh, mr.Path("C:/Users/micah/Desktop/ponts 2a/PROJET IMI/mesh.stl"))'''

slice_spacing = 12.0

volume = np.stack(image_sequence, axis=-1)
verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
verts[:, 2] *= slice_spacing
mesh = meshio.Mesh(points=verts, cells=[("triangle", faces)])
meshio.write("C:/Users/micah/Desktop/ponts 2a/PROJET IMI/ankle_mesh.stl", mesh, file_format="stl")

# Perform connected component analysis on the mesh
labels = measure.label(volume > 0.5)

# Count the number of unique labels (connected components)
num_labels = np.max(labels) + 1

# Generate random colors for each label
label_colors = np.random.rand(num_labels, 3)

# Create an array to hold the colors for each face
face_colors = np.zeros((faces.shape[0], 3), dtype=np.float64)

# Assign colors to faces based on the label of the connected component they belong to
for label in range(num_labels):
    face_indices = np.where(labels == label)[0]
    face_colors[face_indices] = label_colors[label]


# Create a trimesh object with vertices, faces, and face colors
mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors)


# Save the mesh as a PLY file with colors
output_file = "C:/Users/micah/Desktop/ponts 2a/PROJET IMI/colored_mesh.ply"
mesh.export(output_file)






# Prepare the vertices, faces, and vertex colors
vertices = verts.tolist()
faces = np.asarray(faces).flatten()
vertex_colors = face_colors.tolist()

# Reshape the vertex and color arrays to one dimension
vertex = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]).reshape(-1)
color = np.array(vertex_colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]).reshape(-1)

# Define custom field names for the face element
face_fields = [('vertex_indices', 'int32', (3,))]

# Create the PLY data structure
vertex_element = PlyElement.describe(vertex, 'vertex')
face_element = PlyElement(face_fields, count=len(faces))
face_element.data = [tuple(faces[i:i+3]) for i in range(0, len(faces), 3)]
color_element = PlyElement.describe(color, 'color')

# Create the PlyData object
ply_data = PlyData([vertex_element, face_element, color_element])


# Save the colored mesh as a PLY file
output_file = "C:/Users/micah/Desktop/ponts 2a/PROJET IMI/colored_mesh.ply"
ply_data.write(output_file)


