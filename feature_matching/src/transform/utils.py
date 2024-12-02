import torch.nn.functional as F
import torch

def affine2theta(affine, shape):
    h, w = shape[0], shape[1]
    temp = affine
    theta = torch.zeros_like(affine)
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1]*h/w
    theta[0, 2] = temp[0, 2]*2/w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0]*w/h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2]*2/h + theta[1, 0] + theta[1, 1] - 1
    return theta

def tensor_affine_transform(tensor, tensor_transform):
    affine_grid = F.affine_grid(tensor_transform, tensor.size())
    transformed_tensor = F.grid_sample(tensor, affine_grid)
    return transformed_tensor
