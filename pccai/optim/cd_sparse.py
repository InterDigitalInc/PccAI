# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Compute Chamfer Distance loss for MinkowskiEngine sparse tensors

import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))
from modules.nnd import NNDModule
nndistance = NNDModule()


class ChamferDistSparse(PccLossBase):
    """Chamfer distance loss for sparse voxels."""

    def __init__(self, loss_args, syntax):
        super().__init__(loss_args, syntax)


    def xyz_loss(self, loss_out, net_in, net_out):
        """Compute the xyz-loss."""

        x_hat = net_out['x_hat']
        gt = net_out['gt']
        batch_size = x_hat[-1][0].round().int().item() + 1
        dist = torch.zeros(batch_size, device=x_hat.device)
        for i in range(batch_size):
            dist_out, dist_x, _, _ = nndistance(
                x_hat[x_hat[:, 0].round().int()==i, 1:].unsqueeze(0).contiguous(), 
                gt[gt[:, 0] == i, 1:].unsqueeze(0).float().contiguous()
            )
            dist[i] = torch.max(torch.mean(dist_out), torch.mean(dist_x))
        loss = torch.mean(dist)
        loss_out['xyz_loss'] = loss.unsqueeze(0) # write the 'xyz_loss' as return


    def loss(self, net_in, net_out):
        """Overall R-D loss computation."""

        loss_out = {}

        # Rate loss
        if 'likelihoods' in net_out and len(net_out['likelihoods']) > 0:
            self.bpp_loss(loss_out, net_out['likelihoods'], net_out['gt'].shape[0])
        else:
            loss_out['bpp_loss'] = torch.zeros((1,))
            if net_out['x_hat'].is_cuda:
                loss_out['bpp_loss'] = loss_out['bpp_loss'].cuda()
        
        # Distortion loss
        self.xyz_loss(loss_out, net_in, net_out)

        # R-D loss = alpha * D + beta * R
        loss_out["loss"] = self.alpha * loss_out['xyz_loss'] +  self.beta * loss_out["bpp_loss"] # R-D loss
        return loss_out
