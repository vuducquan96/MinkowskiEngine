import numpy as np
import pptk
import spconv
from pdb import set_trace as bp
import torch
from m3d import read_bin_file
from copy import copy as cp
from torch import nn
import time

xx, yy, zz = 3, 3, 3
points = []
def init_points():
    points = []
    x, y, z = 0, 0, 0
    while x < xx:
        y = 0
        while y < yy:
            z = 0
            while z < zz:
                points.append([x, y, z])
                z += 0.5
            y += 0.5
        x += 0.5
    points = np.array(points, dtype=np.float32)
    point1 = cp(points)
    point1[:, 0] -= 6
    point2 = cp(points)
    point2[:, 0] += 6
    points = np.concatenate((points, point1, point2), axis=0)

    return points

points = init_points()
voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-10.0, -10.0, -3.0, 10.0, 10.0, 3.0],
    max_num_points=100,
    full_mean=False,
    max_voxels=200000
)

class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net =  spconv.SparseConv3d(32, 64, 3)
        print(type(self.net.weight))
        # self.net.bias = 0.1
        # # print(self.net.weight.shape)
        # print(self.net.bias)
        # print(self.net.bias.)
        # exit()
        self.sub_net = spconv.SparseConv3d(32, 64, 3)
        print(self.sub_net.weight)
        exit()
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x = self.net(x)
        return x

test_net = ExampleNet(shape = [voxel_generator.grid_size[2], voxel_generator.grid_size[1], voxel_generator.grid_size[0]]).cuda()
voxels, coords, num_points_per_voxel = voxel_generator.generate(points)
batch_size = 1
elems = [coords]
coors = []
for i, coor in enumerate(elems):
    coor_pad = np.pad(
        coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
    coors.append(coor_pad)
coors = np.concatenate(coors, axis=0)
voxels = np.concatenate(voxels, axis=0)

now = time.time()
out = test_net(torch.from_numpy(voxels).cuda(), torch.from_numpy(coors).cuda(),1)
print((time.time() - now) / 60)

print(out.dense().shape)
print("max:", torch.min(out.dense()))
#for x in range(-xx, xx, 1):
#    for y in range(-yy, yy, 1):
#        for z in range(-zz, zz, 1):
#            points.append([x,y,z])



#points = read_bin_file("test.bin", 3)
#points = torch.from_numpy(points).to(torch.float32)


print(voxels.shape)
v1 = pptk.viewer(points)
v1.set(point_size=0.05, phi=3.141, theta=0.785)
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
v.set(point_size=0.05, phi=3.141, theta=0.785)

