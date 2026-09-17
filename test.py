import torch

tensor_3d = torch.tensor([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [10, 11, 12]]
])

print(tensor_3d.shape)  # torch.Size([2, 2, 3])