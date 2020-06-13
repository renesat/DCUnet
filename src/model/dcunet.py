#!/usr/bin/env python3

import torch
import torch.nn as nn

from .complexnet import ComplexBatchNorm2d, ComplexConv2d, ComplexReLU


class DCUnetConv(nn.Module):
    iscomplex: bool

    def __init__(self, layers_params, iscomplex: bool = False):
        super(DCUnetConv, self).__init__()

        assert (len(layers_params) % 2 == 0), 'Count of layers not even'

        self.layers = []
        self.iscomplex = iscomplex
        for i, layer_params in enumerate(layers_params):

            if i > 0:
                in_chanels = layers_params[i - 1]['chanels']
                is_have_connect = i - len(layers_params) // 2 > 0
                if is_have_connect:
                    prev_connect_index = len(layers_params) - i
                    in_chanels += layers_params[prev_connect_index]['chanels']
            else:
                in_chanels = 1
            if self.iscomplex:
                layer = self.__create_complex_conv_layer(
                    in_chanels, layer_params)
            else:
                layer = self.__create_conv_layer(in_chanels, layer_params)
            self.layers.append(layer)

    @staticmethod
    def __create_conv_layer(in_chanels: int, params):
        conv = nn.Conv2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        bn = nn.BatchNorm2d(params['chanels'])
        f = nn.ReLU()
        return nn.Sequential(conv, bn, f)

    @staticmethod
    def __create_complex_conv_layer(in_chanels: int, params):
        conv = ComplexConv2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        bn = ComplexBatchNorm2d(params['chanels'])
        f = ComplexReLU()
        return nn.Sequential(conv, bn, f)

    def forward(self, x):
        """
        x.shape = (B, F, W, H, 2) if iscomlex else  (B, F, W, H)
        """
        outs = []
        for i, layer in enumerate(self.layers):
            is_have_connect = i - len(self.layers) // 2 > 0
            if is_have_connect:
                prev_connect_index = len(self.layers) - i
                x = torch.cat((
                    outs[prev_connect_index],
                    x,
                ), dim=2)
            x = layer(x)
        return x
