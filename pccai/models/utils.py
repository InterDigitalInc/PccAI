# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Elementary modules and utility functions to process point clouds

import numpy as np
import torch
import torch.nn as nn


def get_Conv2d_layer(dims, kernel_size, stride, doLastRelu):
    """Elementary 2D convolution layers."""

    layers = []
    for i in range(1, len(dims)):
        padding = int((kernel_size - 1) / 2) if kernel_size != 1 else 0
        layers.append(nn.Conv2d(in_channels=dims[i-1], out_channels=dims[i],
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU(inplace=True))
    return layers # nn.Sequential(*layers)


class Conv2dLayers(nn.Sequential):
    """2D convolutional layers.

    Args:
        dims: dimensions of the channels
        kernel_size: kernel size of the convolutional layers.
        doLastRelu: do the last Relu (nonlinear activation) or not.
    """
    def __init__(self, dims, kernel_size, doLastRelu=False):
        layers = get_Conv2d_layer(dims, kernel_size, 1, doLastRelu) # Note: may need to init the weights and biases here
        super(Conv2dLayers, self).__init__(*layers)


def get_and_init_FC_layer(din, dout, init_bias='zeros'):
    """Get a fully-connected layer."""

    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    if init_bias == 'uniform':
        nn.init.uniform_(li.bias)
    elif init_bias == 'zeros':
        li.bias.data.fill_(0.)
    else:
        raise 'Unknown init ' + init_bias
    return li


def get_MLP_layers(dims, doLastRelu, init_bias='zeros'):
    """Get a series of MLP layers."""

    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i], init_bias=init_bias))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    """PointwiseMLP layers.

    Args:
        dims: dimensions of the channels
        doLastRelu: do the last Relu (nonlinear activation) or not.
        Nxdin ->Nxd1->Nxd2->...-> Nxdout
    """
    def __init__(self, dims, doLastRelu=False, init_bias='zeros'):
        layers = get_MLP_layers(dims, doLastRelu, init_bias)
        super(PointwiseMLP, self).__init__(*layers)


class GlobalPool(nn.Module):
    """BxNxK -> BxK"""

    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3) #Bx1xNxK
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)   #BxK
        return X


class PointNetGlobalMax(nn.Sequential):
    """BxNxdims[0] -> Bxdims[-1]"""

    def __init__(self, dims, doLastRelu=False):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveMaxPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalMax, self).__init__(*layers)


class PointNetGlobalAvg(nn.Sequential):
    """BxNxdims[0] -> Bxdims[-1]"""

    def __init__(self, dims, doLastRelu=True):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveAvgPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalAvg, self).__init__(*layers)


class PointNet(nn.Sequential):
    """Vanilla PointNet Model.

    Args:
        MLP_dims: dimensions of the pointwise MLP
        FC_dims: dimensions of the FC to process the max pooled feature
        doLastRelu: do the last Relu (nonlinear activation) or not.
        Nxdin ->Nxd1->Nxd2->...-> Nxdout
    """
    def __init__(self, MLP_dims, FC_dims, MLP_doLastRelu):
        assert(MLP_dims[-1]==FC_dims[0])
        layers = [
            PointNetGlobalMax(MLP_dims, doLastRelu=MLP_doLastRelu),#BxK
        ]
        layers.extend(get_MLP_layers(FC_dims, False))
        super(PointNet, self).__init__(*layers)