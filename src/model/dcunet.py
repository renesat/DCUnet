#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complexnet import (ComplexBatchNorm2d, ComplexConv2d,
                         ComplexConvTranspose2d, ComplexReLU)


class DCUnet(nn.Module):
    iscomplex: bool

    def __init__(self, layers_params, iscomplex: bool = False):
        super(DCUnet, self).__init__()

        assert (len(layers_params) % 2 == 0), 'Count of layers not even'

        self.layers = []
        self.iscomplex = iscomplex
        for i, layer_params in enumerate(layers_params):
            is_have_connect = i >= len(layers_params) // 2 + 1
            is_decoder = i >= len(layers_params) // 2
            if i > 0:
                in_chanels = layers_params[i - 1]['chanels']
                if is_have_connect:
                    prev_connect_index = len(layers_params) - i - 1
                    in_chanels += layers_params[prev_connect_index]['chanels']
            else:
                in_chanels = 1

            if i != len(layers_params) - 1:
                batch_norm = True
                activation = True
            else:
                batch_norm = False
                activation = False

            if not is_decoder:
                if self.iscomplex:
                    layer_type = "complex_encoder"
                else:
                    layer_type = "encoder"
            else:
                if self.iscomplex:
                    layer_type = "complex_decoder"
                else:
                    layer_type = "decoder"

            layer = self.__create_block(
                layer_type,
                in_chanels,
                layer_params['chanels'],
                layer_params['kernel'],
                layer_params['stride'],
                batch_norm=batch_norm,
                activation=activation,
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    @staticmethod
    def __create_block(block_type,
                       in_chanels: int,
                       out_chanels: int,
                       kernel,
                       stride,
                       batch_norm: bool = True,
                       activation: bool = True):
        convType, bnType, fType = {
            "encoder": (
                nn.Conv2d,
                nn.BatchNorm2d,
                nn.ReLU,
            ),
            "decoder": (
                nn.ConvTranspose2d,
                nn.BatchNorm2d,
                nn.ReLU,
            ),
            "complex_encoder": (
                ComplexConv2d,
                ComplexBatchNorm2d,
                ComplexReLU,
            ),
            "complex_decoder": (
                ComplexConvTranspose2d,
                ComplexBatchNorm2d,
                ComplexReLU,
            ),
        }[block_type]
        layers = []
        conv = convType(
            in_chanels,
            out_chanels,
            kernel,
            stride=stride,
        )
        layers.append(conv)

        if batch_norm:
            bn = bnType(out_chanels)
            layers.append(bn)

        if activation:
            f = fType()
            layers.append(f)

        return nn.Sequential(*layers)

    def __interpolate(self, x, width, height):
        def _interpolate(item):
            return F.interpolate(item, size=(width, height))

        if self.iscomplex:
            return torch.stack(tuple(map(_interpolate, x.unbind(4))), dim=4)
        else:
            return _interpolate(x)

    def forward(self, x):
        """
        x.shape = (B, F, W, H, 2) if iscomlex else  (B, F, W, H)
        """
        input_size = x.shape[2:4]
        outs = []
        for i, layer in enumerate(self.layers):
            is_have_connect = i >= len(self.layers) // 2 + 1
            if is_have_connect:
                prev_connect_index = len(self.layers) - i - 1
                connect_x = outs[prev_connect_index]
                prev_x = self.__interpolate(x, *(connect_x.shape[2:4]))
                x = torch.cat((connect_x, prev_x), dim=1)

            x = layer(x)
            if not is_have_connect:
                outs.append(x.data)

        x = self.__interpolate(x, *(input_size))
        return x
