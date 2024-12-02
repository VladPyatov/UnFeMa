import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def ncc_global(source, target):
    size = source.shape[-2] * source.shape[-1]
    source_mean = torch.mean(source, dim=(1, 2, 3), keepdim=True)
    target_mean = torch.mean(target, dim=(1, 2, 3), keepdim=True)
    source_std = torch.std(source, dim=(1, 2, 3))
    target_std = torch.std(target, dim=(1, 2, 3))
    ncc = (1/size)*torch.sum((source - source_mean)*(target-target_mean), dim=(1, 2, 3)) / (source_std * target_std)
    return -ncc.mean()


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out



class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def gram_loss(x, y):
    return F.mse_loss(gram_matrix(x), gram_matrix(y))


class PerceptualLoss:
    def __init__(self, scales=[1], num_channels=1, loss_weights=[1,1,1,1,1]):
        self.scales = scales
        self.loss_weights = loss_weights
        self.pyramid = ImagePyramide(self.scales, num_channels)
        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
            self.vgg = self.vgg.cuda()
    #@torch.no_grad()
    def __call__(self, x, y, device=None):
        #x, y = x.clone(), y.clone()
        h, w = x.shape[-2:]
        mask_h, mask_w = int(0.1 * h), int(0.1 * w)
        x[...,:mask_h, :] = 0
        x[...,-mask_h:, :] = 0
        y[...,:mask_h, :] = 0
        y[...,-mask_h:, :] = 0
        x[...,:mask_w] = 0
        x[...,-mask_w:] = 0
        y[...,:mask_w] = 0
        y[...,-mask_w:] = 0
        if len(x.shape) == 2:
            x = x[None, None, ...]
        if len(y.shape) == 2:
            y = y[None, None, ...]
        
        pyramid_x = self.pyramid(x)
        pyramid_y = self.pyramid(y)
        value_total = 0
        for scale in self.scales:
            x_vgg = self.vgg(pyramid_x['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramid_y['prediction_' + str(scale)])
            for i, weight in enumerate(self.loss_weights):
                value = ncc_global(x_vgg[i].transpose(0, 1),  y_vgg[i].transpose(0, 1))#torch.abs(x_vgg[i] - y_vgg[i]).mean()
                print(value)
                value_total += weight * value
        return value_total
