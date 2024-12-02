"""Module containing coordinate transformations"""
import numpy as np
import torch
import torch.nn as nn


def _center_and_normalize_points(points: torch.Tensor):
    """Center and normalize image points.

    Parameters
    ----------
    points : (B, N, D) array
        The coordinates of the image points.
    Returns
    -------
    matrix : (B, 1, D+1, D+1) array
        The transformation matrix to obtain the new points.
    new_points : (B, N, D) array
        The transformed image points.
    """
    b_size, n, d = points.shape
    centroid = torch.mean(points, dim=1, keepdim=True)

    centered = points - centroid
    rms = torch.sqrt(torch.sum(centered ** 2, dim=(1, 2)) / n).unsqueeze(-1)

    mask = rms == 0
    if any(mask):
        rms = rms.masked_fill(mask, 1)

    norm_factor = np.sqrt(d) / rms
    eye = torch.eye(d, device=points.device).unsqueeze(0).unsqueeze(0).repeat(b_size, 1, 1, 1)
    part_matrix = norm_factor[..., None, None] * torch.cat((eye, -centroid.unsqueeze(-1)), axis=-1)
    zeros = torch.zeros((b_size, 1, 1, 3), device=points.device)
    zeros[..., -1] += 1
    matrix = torch.cat((part_matrix, zeros), axis=-2)

    points_h = torch.cat((points.transpose(1,2), torch.ones((b_size,1,n), device=points.device)), axis=1)

    new_points_h = (matrix[:,0] @ points_h).transpose(1,2)
    new_points = new_points_h[..., :d] / new_points_h[..., d:]

    if any(mask):
        print('transforms - CENTER ERROR')
        matrix = matrix.masked_fill(mask[..., None, None], torch.nan)
        new_points = new_points.masked_fill(mask[..., None], torch.nan)

    return matrix, new_points


def transform(points, transform_matrix):
    """Transform points
    
    Parameters
        ----------
        points : (B, N, 2) tensor
            Source coordinates.
        transform_matrix : (B, 1, 3, 3) tensor
            Transformation matrix.
        Returns
        -------
        t_points : (B, N, 2) tensor
            Transformed points.
    """
    b_size, n, d = points.shape
    device = points.device

    points_h = torch.cat((points, torch.ones((b_size, n, 1), device=device)), dim=-1)
    t_points_h = transform_matrix[:, 0] @ points_h.transpose(-1, -2)
    t_points_h = t_points_h.transpose(-1, -2)
    t_points = t_points_h[..., :2] / t_points_h[..., [2]]
    
    return t_points


def residuals(src, dst, transform_matrix=None):
    """Determine residuals of transformed destination coordinates.
        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.
        Parameters
        ----------
        src : (B, N, 2) tensor
            Source coordinates.
        dst : (B, N, 2) tensor
            Destination coordinates.
        transform_matrix : (B, 1, 3, 3) tensor
            Transformation matrix.
        Returns
        -------
        residuals : (B, N) tensor
            Residual for coordinate.
        """
    if transform_matrix is not None:
        src = transform(src, transform_matrix)
    return torch.sqrt(torch.sum((src - dst) ** 2, dim=-1))


class AffineTransform2D(nn.Module):
    """
    Differentiable implementation of skimage's affine transformation. 
    Transform estimation implemented in forward() method.
    For details, check https://scikit-image.org/docs/stable/api/skimage.transform.html#affinetransform
    """
    def __init__(self, cpu_computation=False, full_svd=True, limit_of_points=None) -> None:
        super().__init__()
        self.cpu_computation = cpu_computation
        self.full_svd = full_svd
        self.limit = limit_of_points
    
    def forward(self, src, dst, weights=None):
        """Estimate transform from corresponding source (src) and destination (dst) points.

        Parameters
        ----------
        src : (B, N, 2) tensor
            Source coordinates.
        dst : (B, N, 2) tensor
            Destination coordinates.
        weights : (B, N) tensor, optional
            Relative weight values for each pair of points.
        Returns
        -------
        transformation : (B, 1, 3, 3) tensor
            Estimated transformation
        """
        device = src.device
        if src.shape != dst.shape:
            #raise AssertionError(src.shape)
            return None
        if src.shape[1] < 2:
            #raise AssertionError(src.shape)
            return None
        if self.limit is not None and src.shape[1] > self.limit:
            indices = torch.randint(src.shape[1], (self.limit,), device=device)
            src = src[:, indices]
            dst = dst[:, indices]
            weights = weights[:, indices]
        b_size, n, d = src.shape
        device = src.device

        src_matrix, src_norm = _center_and_normalize_points(src)
        dst_matrix, dst_norm = _center_and_normalize_points(dst)

        src_x, src_y = torch.chunk(src_norm, dim=-1, chunks=2)
        dst_x, dst_y = torch.chunk(dst_norm, dim=-1, chunks=2)
        ones, zeros = torch.ones_like(src_x), torch.zeros_like(src_x)

        A_x = torch.cat((src_x, src_y, ones, zeros, zeros, zeros, dst_x), dim=-1)
        A_y = torch.cat((zeros, zeros, zeros,src_x, src_y, ones, dst_y), dim=-1)
        # B x 2N x 7
        A = torch.cat((A_x,A_y), dim=-2)

        if weights is not None:
            weights_norm = torch.sqrt(weights / torch.amax(weights, dim=1, keepdim=True))
            weights_tiled = torch.tile(weights_norm, (1, d))[..., None]
            eye = torch.eye(d * n, device=device).unsqueeze(0).repeat(b_size, 1, 1)
            W = eye * weights_tiled
            A = W @ A

        # cpu computation is a bit stable
        if self.cpu_computation:
            right_singular_vector = []
            for i in range(b_size):
                try:
                    _, _, i_Vh = torch.linalg.svd(A[i].cpu(), full_matrices=self.full_svd)
                    i_Vh_last = i_Vh[[-1]].to(device)
                except Exception as error:
                    i_Vh_last = torch.tensor([-0.] * (A[i].shape[-1] - 1) + [-1.],
                                            device=A.device, requires_grad=A.requires_grad)
                    i_Vh_last = i_Vh_last.view(1, -1)
                finally:
                    right_singular_vector.append(i_Vh_last)
            right_singular_vector = torch.cat(right_singular_vector).unsqueeze(1)
        else:
            _, _, Vh = torch.linalg.svd(A, full_matrices=self.full_svd)
            right_singular_vector = Vh[:,[-1]]
        
        # right_singular_vector: B x 1 x 7
        # for badly conditioned (~unsolvable) systems right_singular_vector is just [0., 0., 0., 0., 0., 0., -1.]
    
        H = - right_singular_vector[..., :-1] / right_singular_vector[..., [-1]]
        H = H.view(-1, 2, 3)
        third_row = torch.cat((torch.zeros((b_size, 1, 2), device=device),
                                torch.ones((b_size, 1, 1), device=device)),
                                dim=-1)
        H = torch.cat((H, third_row), dim=-2).unsqueeze(1)

        H = torch.linalg.solve(dst_matrix, H) @ src_matrix
        
        transformation = H / H[..., [-1], [-1]].unsqueeze(1)
        if transformation.isnan().any():
            print('transformation ERROR')
        return transformation
        