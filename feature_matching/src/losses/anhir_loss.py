import math

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from src.transform.transforms import AffineTransform2D, residuals
from src.transform.utils import affine2theta, tensor_affine_transform
from .perceptual_loss import PerceptualLoss

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3, kernel=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.kernel = kernel
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=[win[0], win[1]]))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i].loss(I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool2d(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = nn.functional.avg_pool2d(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)


def ncc_global(source, target):
    size = source.shape[-2] * source.shape[-1]
    source_mean = torch.mean(source, dim=(1, 2, 3), keepdim=True)
    target_mean = torch.mean(target, dim=(1, 2, 3), keepdim=True)
    source_std = torch.std(source, dim=(1, 2, 3))
    target_std = torch.std(target, dim=(1, 2, 3))
    ncc = (1/size)*torch.sum((source - source_mean)*(target-target_mean), dim=(1, 2, 3)) / (source_std * target_std)
    return -ncc.mean()

def batch_loss(source, target, b_ids, kpts0, kpts1, weights, loss_type, gt_kpts=[None, None]):
    if loss_type == 'ncc':
        loss_func = ncc_global
    elif loss_type == 'ncc_local':
        loss_func = NCC(win=[32, 32]).loss
    elif loss_type == 'ncc_local_multi':
        loss_func = multi_resolution_NCC(win=[128, 128])
    elif loss_type == 'l1':
        loss_func = nn.MSELoss()
    elif loss_type == 'l2':
        loss_func = nn.L1Loss()
    elif loss_type == 'perc':
        loss_func = PerceptualLoss()
    loss = []
    transform = AffineTransform2D(limit_of_points=5555)
    warped_images = np.zeros(source.shape, dtype=np.int32)
    residual_errors = [] if gt_kpts[0] is not None else None
    for b_idx in b_ids.unique():
        b_mask = b_idx == b_ids
        b_source = source[b_idx][None]
        b_target = target[b_idx][None]
        b_kpts0 = kpts0[b_mask][None]
        b_kpts1 = kpts1[b_mask][None]
        b_weights = weights[b_mask][None]
        # find transform from target to source
        affine_matrix = transform(b_kpts1, b_kpts0, b_weights)
        if affine_matrix is None or affine_matrix.isnan().any():
            continue
        if gt_kpts[0] is not None:
            error = residuals(gt_kpts[1], gt_kpts[0], affine_matrix).median() / math.sqrt(b_source.shape[-1] ** 2 +  b_source.shape[-2] ** 2)
            residual_errors.append(error)
        theta_matrix = affine2theta(affine_matrix[0, 0, :2], b_source.shape[2:])[None]
        # kornia implementation?
        # affine_matrix = transform(b_kpts0, b_kpts1, b_weights)[:, 0, :2]
        # _, _, h, w = b_source.shape
        # b_source_warped = K.geometry.warp_affine(b_source, affine_matrix, dsize=(h, w), align_corners=False)
        b_source_warped = tensor_affine_transform(b_source, theta_matrix)
        warped_images[b_idx] = (b_source_warped[0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.int32)
        loss.append(loss_func(b_source_warped, b_target))
    #print(loss)

    loss = torch.stack(loss).mean() if len(loss) != 0 else None
    
    if gt_kpts[0] is not None:
        residual_errors = torch.stack(residual_errors).mean() if len(residual_errors) != 0 else None

    return loss, warped_images, residual_errors

class TissueLoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.coarse_type = self.loss_config['coarse_type']
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, data):
        source, target = data['image0'], data['image1']
        b_ids = data['b_ids']
        kpts0, kpts1 = data['mkpts0_c'], data['mkpts1_c']
        weights = data['mconf']
        gt_kpts = [data.get('image0_kpts', None), data.get('image1_kpts', None)]
        if self.coarse_type == 'ransac':
            raise NotImplementedError()
        else:
            return batch_loss(source, target, b_ids, kpts0, kpts1, weights, self.coarse_type, gt_kpts)
        
    def compute_fine_loss(self, data):
        if self.fine_type is None:
            return None, None, None
        
        source, target = data['image0'], data['image1']
        b_ids = data['b_ids']
        kpts0, kpts1 = data['mkpts0_f'], data['mkpts1_f']
        weights = data['mconf']
        gt_kpts = [data.get('image0_kpts', None), data.get('image1_kpts', None)]
        
        if self.fine_type == 'ransac':
            raise NotImplementedError()
        else:
            return batch_loss(source, target, b_ids, kpts0, kpts1, weights, self.fine_type, gt_kpts)

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        loss = 0
        # 1. coarse-level loss
        loss_c, warped_c, residual_errors_c = self.compute_coarse_loss(data)
        if loss_c is not None:
            loss = loss_c * self.loss_config['coarse_weight']
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        else:
            loss_scalars.update({'loss_c': torch.tensor(1.)})
        if residual_errors_c is not None:
            loss_scalars.update({'residual_errors_c': residual_errors_c.clone().detach().cpu()})
        else:
            loss_scalars.update({'residual_errors_c': torch.tensor(1.)})

        # 2. fine-level loss
        loss_f, warped_f, residual_errors_f = self.compute_fine_loss(data)
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound
        if residual_errors_f is not None:
            loss_scalars.update({'residual_errors_f': residual_errors_f.clone().detach().cpu()})
        else:
            loss_scalars.update({'residual_errors_f': torch.tensor(1.)})

        if loss == 0:
            print(f'No loss: {loss}, loss_c: {loss_c}, loss_f: {loss_f}')
            data['zero_loss'] = True
            # no actual backward
            loss = 0 * data['conf_matrix'].mean()
        
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars, "image01_c": warped_c, "image01_f": warped_f})
