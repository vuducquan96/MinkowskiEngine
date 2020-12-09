import numpy as np
import pptk
import spconv
from pdb import set_trace as bp
import torch
from m3d import read_bin_file
from copy import copy as cp

xx, yy, zz = 3, 3, 3
points = []
voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-80.0, -80.0, -3.0, 80.0, 80.0, 3.0],
    max_num_points=10,
    full_mean=False,
    max_voxels=200000
)

#for x in range(-xx, xx, 1):
#    for y in range(-yy, yy, 1):
#        for z in range(-zz, zz, 1):
#            points.append([x,y,z])
x,y,z = 0, 0, 0
while x < xx:
    y = 0
    while y < yy:
        z = 0
        while z < zz:
            points.append([x,y,z])
#            points.append([x,y,z])
#            points.append([x,y,z])
#            points.append([x,y,z])
#            points.append([x,y,z])
            z += 0.5
        y += 0.5
    x += 0.5

#points = np.array(points, dtype=np.float32)
points = np.array(points, dtype=np.float32)

point1 = cp(points)
point1[:, 0] -= 6
point2 = cp(points)
point2[:, 0] += 6

points = np.concatenate((points, point1, point2), axis=0)
#points = read_bin_file("test.bin", 3)
#points = torch.from_numpy(points).to(torch.float32)

voxels, coords, num_points_per_voxel = voxel_generator.generate(points)
print(voxels.shape)
v = pptk.viewer(points)
v.set(point_size=0.01, phi=3.141, theta=0.785)
#bp()
for x in range(coords.shape[0]):
    tt = coords[x,2]
    coords[x,2] = coords[x,0]
    coords[x,0] = tt

#bp()
#x = coords[:,0]
#y = coords[:,1]
#z = coords[:,2]
#coords[:,0] = z
#coords[:,1] = x
#coords[:,2] = y
#bp()

v = pptk.viewer(coords)
v.set(point_size=0.01, phi=3.141, theta=0.785)

