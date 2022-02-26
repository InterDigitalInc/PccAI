# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import math
import torch
import sys
import os

class PccLossBase:
    """A base class of rate-distortion loss computation for point cloud compression."""

    def __init__(self, loss_args, syntax):
        self.alpha = loss_args['alpha']
        self.beta = loss_args['beta']
        self.hetero = syntax.hetero
        self.phase = syntax.phase


    @staticmethod
    def bpp_loss(loss_out, likelihoods, count):
        """Compute the rate loss with the likelihoods."""

        bpp_loss = 0
        for k, v in likelihoods.items():
            if v is not None:
                loss = torch.log(v).sum() / (-math.log(2) * count)
                bpp_loss += loss
                loss_out[f'bpp_loss_{k}'] = loss.unsqueeze(0)
        loss_out['bpp_loss'] = bpp_loss.unsqueeze(0)


    def xyz_loss(self, **kwargs):
        """Needs to implement the xyz_loss"""

        raise NotImplementedError()


    def loss(self, **kwargs):
        """Needs to implement the overall loss. Can be R-D loss for lossy compression, or rate-only loss for lossless compression."""

        raise NotImplementedError()
