import numpy as np
import pptk
import spconv
from pdb import set_trace as bp
import torch
from m3d import read_bin_file
from copy import copy as cp
from torch import nn
import time
from torch import nn
from torch.nn import functional as F

points = []


class SimpleVoxel(nn.Module):
    def __init__(self, num_input_features=3):
        super(SimpleVoxel, self).__init__()
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]

        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()


def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte-minimum) / (maximum - minimum)
    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

def kernel_to_points(kernel, numb):
    points = []
    data = kernel[0,numb, :,:,:]
    xx, yy, zz = data.shape
    power = []
    for x in range(xx):
        for y in range(yy):
            for z in range(zz):
                if data[x,y,z] != 0:
                    points.append([z,x,y])
                    power.append(data[x,y,z])

    points = np.array(points, dtype=np.float32)
    colors = inte_to_rgb(power)

    k = pptk.viewer(points)
    k.attributes(colors)
    k.set(point_size=0.03, phi=3.141, theta=0.785)

def init_points():
    points = []
    x, y, z = 0, 0, 0
    xx, yy, zz = 2, 2, 2
    delta = 0.08

    while x < xx:
        y = 0
        while y < yy:
            z = 0
            while z < zz:
                if x > y:
                    points.append([x, y, z])
                z += delta
            y += delta
        x += delta
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
    max_num_points=3,
    full_mean=False,
    max_voxels=200000
)

class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.net =  spconv.SparseConv3d(32, 64, 3)
#        print(type(self.net.weight))

        # self.net.bias = 0.1
        # # print(self.net.weight.shape)
        # print(self.net.bias)
        # print(self.net.bias.)
        # exit()
        self.sub_net = spconv.SparseSequential(
                spconv.SubMConv3d(32, 64, 3))
        
#        self.sub_net_1 = spconv.SubMConv3d(32, 128, 3, bias=False, indice_key=None)
#
#        self.dancing = spconv.ToDense()
#        print(self.sub_net.weight)
#        print(self.sub_net.weight.shape)
#        bp()
#        exit()
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
#        bp()
#        x = self.net(x)
        x = self.sub_net(x)
#        x = self.net(x)

#        x = self.dancing(x)
        # bp()
        return x

device = "cuda"
test_net = ExampleNet(shape = [voxel_generator.grid_size[2], voxel_generator.grid_size[1], voxel_generator.grid_size[0]]).to(device)

mean_voxel = SimpleVoxel().to(device)
voxels, coords, num_points_per_voxel = voxel_generator.generate(points)
voxels = torch.from_numpy(voxels).to(device)
num_points_per_voxel = torch.from_numpy(num_points_per_voxel).to(device)
voxels_feature = mean_voxel(voxels, num_points_per_voxel)
#bp()
batch_size = 1
elems = [coords]
coors = []
for i, coor in enumerate(elems):
    coor_pad = np.pad(
        coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
    coors.append(coor_pad)
coors = np.concatenate(coors, axis=0)

now = time.time()
out = test_net(voxels_feature, torch.from_numpy(coors).to(device), 1)
bp()
print((time.time() - now) / 60)
#print(out.dense().shape)
#print("max:", torch.min(out.dense()))
kernel_to_points(out.dense().cpu().detach().numpy(), 0)
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

