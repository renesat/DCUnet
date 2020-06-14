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
            else:
                batch_norm = False

            if not is_decoder:
                if self.iscomplex:
                    layer = self.__create_complex_conv_layer(
                        in_chanels, layer_params, batch_norm=batch_norm)
                else:
                    layer = self.__create_conv_layer(in_chanels,
                                                     layer_params,
                                                     batch_norm=batch_norm)
            else:
                if self.iscomplex:
                    layer = self.__create_complex_deconv_layer(
                        in_chanels, layer_params, batch_norm=batch_norm)
                else:
                    layer = self.__create_deconv_layer(in_chanels,
                                                       layer_params,
                                                       batch_norm=batch_norm)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    @staticmethod
    def __create_conv_layer(in_chanels: int, params, batch_norm: bool = True):
        conv = nn.Conv2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        f = nn.ReLU()
        if batch_norm:
            bn = nn.BatchNorm2d(params['chanels'])
            seq = nn.Sequential(conv, bn, f)
        else:
            seq = nn.Sequential(conv, f)
        return seq

    @staticmethod
    def __create_deconv_layer(in_chanels: int,
                              params,
                              batch_norm: bool = True):
        conv = nn.ConvTranspose2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        f = nn.ReLU()
        if batch_norm:
            bn = nn.BatchNorm2d(params['chanels'])
            seq = nn.Sequential(conv, bn, f)
        else:
            seq = nn.Sequential(conv, f)
        return seq

    @staticmethod
    def __create_complex_conv_layer(in_chanels: int,
                                    params,
                                    batch_norm: bool = True):
        conv = ComplexConv2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        f = ComplexReLU()
        if batch_norm:
            bn = ComplexBatchNorm2d(params['chanels'])
            seq = nn.Sequential(conv, bn, f)
        else:
            seq = nn.Sequential(conv, f)
        return seq

    @staticmethod
    def __create_complex_deconv_layer(in_chanels: int,
                                      params,
                                      batch_norm: bool = True):
        conv = ComplexConvTranspose2d(
            in_chanels,
            params['chanels'],
            params['kernel'],
            stride=params['stride'],
        )
        f = ComplexReLU()
        if batch_norm:
            bn = ComplexBatchNorm2d(params['chanels'])
            seq = nn.Sequential(conv, bn, f)
        else:
            seq = nn.Sequential(conv, f)
        return seq

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
                if self.iscomplex:
                    prev_x = torch.stack((
                        F.interpolate(x.unbind(4)[0],
                                      size=connect_x.shape[2:4]),
                        F.interpolate(x.unbind(4)[1],
                                      size=connect_x.shape[2:4]),
                    ),
                                         dim=4)
                else:
                    prev_x = F.interpolate(x, size=connect_x.shape[2:4])
                x = torch.cat((connect_x, prev_x), dim=1)

            x = layer(x)
            if not is_have_connect:
                outs.append(x.data)

        if self.iscomplex:
            x = torch.stack((
                F.interpolate(x.unbind(4)[0], size=input_size),
                F.interpolate(x.unbind(4)[1], size=input_size),
            ),
                            dim=4)
        else:
            x = F.interpolate(x, size=input_size)

        return x
