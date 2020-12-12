# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import numpy as np

import torch
import MinkowskiEngine as ME
from pdb import set_trace as bp
import spconv
from copy import copy as cp

def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j])
    return np.array(coords)

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

device = "cuda"
def data_loader(nchannel=3,
                max_label=5,
                is_classification=True,
                seed=-1,
                batch_size=2):
    if seed >= 0:
        torch.manual_seed(seed)

    # points = init_points()
    # voxel_size = 0.1
    # sinput = ME.SparseTensor(
    #     feats=torch.from_numpy(colors).float(),
    #     coords=ME.utils.batched_coordinates([points / voxel_size]),
    #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    # ).to(device)

    data = [
        "   X   ",  #
        "  X X  ",  #
        " XXXXX "
    ]

    # Generate coordinates
    # bp()
    coords = [get_coords(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.randn(N, nchannel)
    label = (torch.rand(2 if is_classification else N) * max_label).long()
    return coords, feats, label
