import torch
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor, MinkowskiConvolution

in_channels, out_channels, D = 2, 3, 2
coords1 = torch.IntTensor([[0, 0], [0, 1], [1, 1]])
feats1 = torch.DoubleTensor([[1, 2], [3, 4], [5, 6]])

coords2 = torch.IntTensor([[1, 1], [1, 2], [2, 1]])
feats2 = torch.DoubleTensor([[7, 8], [9, 10], [11, 12]])
# coords, feats = ME.utils.sparse_collate([coords1, coords2], [feats1, feats2])
coords, feats = ME.utils.sparse_collate([coords1], [feats1])
input = SparseTensor(feats, coords=coords)
input.requires_grad_()
dinput, min_coord, tensor_stride = input.dense()

# Initialize context
conv = MinkowskiConvolution(in_channels,out_channels,kernel_size=3,stride=2,has_bias=True,dimension=D)
conv = conv.double()
output = conv(input)
print(input.C)
print("===")
print(output.C)
print("===")

# Convert to a dense tensor
dense_output, min_coord, tensor_stride = output.dense()
print(dense_output.shape)