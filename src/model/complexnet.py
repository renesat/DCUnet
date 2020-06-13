#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__()

        self.conv_real = nn.Conv2d(*args, **kwargs)
        self.conv_image = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        """
        x.shape = (batch_size, in_chanel, width, height, 2)
        """
        conv0 = self.conv_real(x.unbind(4)[0])
        conv1 = self.conv_image(x.unbind(4)[1])
        real_part = (conv0 - conv1).unsqueeze(4)
        image_part = (conv0 + conv1).unsqueeze(4)
        return torch.cat((
            real_part,
            image_part,
        ), dim=4)


class ComplexReLU(nn.Module):
    """
    CReLU(x + i y)= ReLU(x) + i ReLU(y)

    In code is base ReLU.
    """
    def __init__(self, *args, **kwargs):
        super(ComplexReLU, self).__init__()
        self.f = nn.ReLU(*args, **kwargs)

    def forward(self, x):
        """
        x.shape = (batch_size, in_chanel, width, height, 2)
        """
        return self.f(x)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-05):
        super(ComplexBatchNorm2d, self).__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.beta = nn.Parameter(torch.zeros(num_features, 2))
        self.gamma = nn.Parameter(torch.empty(num_features, 2, 2))
        self.gamma[:, 0, 0].fill_(1 / np.sqrt(2))
        self.gamma[:, 1, 1].fill_(1 / np.sqrt(2))
        self.gamma[:, 1, 0].zero_()
        self.gamma[:, 0, 1].zero_()

        self.running_mean = torch.zeros(num_features, 2)
        self.running_var = torch.empty(num_features, 2, 2)
        self.running_var[:, 0, 0].fill_(1 / np.sqrt(2))
        self.running_var[:, 1, 1].fill_(1 / np.sqrt(2))
        self.running_var[:, 1, 0].zero_()
        self.running_var[:, 0, 1].zero_()

    def forward(self, x):
        """
        x.shape = (B, F, W, H, 2)
        """
        axis = (0, 2, 3)

        # Mean value
        if self.training:
            mean_complex = x.mean(dim=axis)  # (F, 2)

            with torch.no_grad():
                self.running_mean *= (1 - self.momentum)
                self.running_mean += self.momentum * mean_complex.data
        else:
            mean_complex = self.running_mean

        # Centered data
        centered_x = x - mean_complex.reshape(1, self.num_features, 1, 1, 2)
        centered_x_real = centered_x.unbind(4)[0]  # (B, F, W, H)
        centered_x_image = centered_x.unbind(4)[1]  # (B, F, W, H)

        # Var value
        if self.training:
            var = x.var(dim=axis, unbiased=False) + self.eps  # (F, 2)
            var_real_real = var.unbind(1)[0]  # (F,)
            var_image_image = var.unbind(1)[1]  # (F,)
            var_real_image = (centered_x_image * centered_x_real).mean(
                dim=axis)  # (F, )

            with torch.no_grad():
                self.running_var *= (1 - self.momentum)
                self.running_var += self.momentum * torch.stack((
                    var_real_real.data,
                    var_real_image.data,
                    var_real_image.data,
                    var_image_image.data,
                )).T.view(self.num_features, 2, 2)
        else:
            var_real_real = self.running_var[:, 0, 0]
            var_image_image = self.running_var[:, 1, 1]
            var_real_image = self.running_var[:, 1, 0]

        # V^(-1/2)
        # Based on: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py [complex_standardization]
        var_det = torch.sqrt(var_real_real * var_image_image -
                             var_real_image * var_real_image)
        t = torch.sqrt(var_real_real + var_image_image + 2 * var_real_image)

        inverse_det_t = 1.0 / (var_det * t)

        var_inv_sqrt_real_real = (var_image_image + var_det) * inverse_det_t
        var_inv_sqrt_real_real = var_inv_sqrt_real_real.reshape(
            1, self.num_features, 1, 1)

        var_inv_sqrt_image_image = (var_real_real + var_det) * inverse_det_t
        var_inv_sqrt_image_image = var_inv_sqrt_image_image.reshape(
            1, self.num_features, 1, 1)

        var_inv_sqrt_real_image = -var_real_image * inverse_det_t
        var_inv_sqrt_real_image = var_inv_sqrt_real_image.reshape(
            1, self.num_features, 1, 1)

        # Get result
        result_x_real = (centered_x_real * var_inv_sqrt_real_real +
                         centered_x_image * var_inv_sqrt_real_image
                         )  # (B, F, W, H)
        result_x_image = (centered_x_real * var_inv_sqrt_real_image +
                          centered_x_image * var_inv_sqrt_image_image
                          )  # (B, F, W, H)

        affine_result_x_real = (
            result_x_real *
            self.gamma[:, 0, 0].reshape(1, self.num_features, 1, 1) +
            result_x_image *
            self.gamma[:, 0, 1].reshape(1, self.num_features, 1, 1)
        )  # (B, F, W, H)
        affine_result_x_image = (
            result_x_real *
            self.gamma[:, 1, 0].reshape(1, self.num_features, 1, 1) +
            result_x_image *
            self.gamma[:, 1, 1].reshape(1, self.num_features, 1, 1)
        )  # (B, F, W, H)

        affine_result_x = torch.stack(
            (affine_result_x_real, affine_result_x_image),
            dim=4) + self.beta.reshape(1, self.num_features, 1, 1, 2)

        return affine_result_x
