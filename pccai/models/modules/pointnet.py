# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# PointNet encoder

import torch
import torch.nn as nn
from pccai.models.utils import PointwiseMLP, GlobalPool
from torch_scatter import scatter_max, scatter_min, scatter_mean


class PointNet(nn.Module):
    """The vanilla PointNet model in homogeneous batching mode.

    Args:
        mlp_dims: Dimension of the MLP
        fc_dims: Dimension of the FC after max pooling
        mlp_dolastrelu: whether do the last ReLu after the MLP
    """

    def __init__(self, net_config, **kwargs):
        super(PointNet, self).__init__()
        self.pointwise_mlp = PointwiseMLP(net_config['mlp_dims'], net_config.get('mlp_dolastrelu', False)) # learnable
        self.fc = PointwiseMLP(net_config['fc_dims'], net_config.get('fc_dolastrelu', False)) # learnable

        # self.pointnet = PointNet(net_config['mlp_dims'], net_config['fc_dims'], net_config['mlp_dolastrelu'])
        self.global_pool = GlobalPool(nn.AdaptiveMaxPool2d((1, net_config['mlp_dims'][-1])))

    def forward(self, data):
        return self.fc(self.global_pool(self.pointwise_mlp(data)))


class PointNetHetero(nn.Module):
    """PointNet in heterogeneous batching mode."""

    def __init__(self, net_config, **kwargs):
        super(PointNetHetero, self).__init__()
        self.pointwise_mlp = PointwiseMLP(net_config['mlp_dims'], net_config.get('mlp_dolastrelu', False)) # learnable
        self.fc = PointwiseMLP(net_config['fc_dims'], False) # learnable
        self.ext_cw = net_config.get('ext_cw', False)

        # Get the syntax
        self.syntax_gt = kwargs['syntax'].syntax_gt
        self.syntax_cw = kwargs['syntax'].syntax_cw

    def forward(self, data):
        device = data.device

        batch_size, pnt_cnt, dims = data.shape[0], data.shape[1], data.shape[2]
        data = data.view(-1, dims)
        block_idx = torch.cumsum(data[:, self.syntax_gt['block_start']] > 0, dim=0) - 1 # compute the block index with cumsum()
        block_idx = block_idx[data[:, self.syntax_gt['block_pntcnt']] > 0] # remove the padding and the skip points
        pc_start = torch.arange(0, batch_size, dtype=torch.long, device=device).repeat_interleave(pnt_cnt)
        pc_start = pc_start[data[:, self.syntax_gt['block_start']] > 0] # remove the "non-start" points
        pc_start = torch.cat((torch.ones(1, device=device), pc_start[1:] - pc_start[0: -1]))
        data = data[data[:, self.syntax_gt['block_pntcnt']] > 0, :] # remove the padding and the skip points

        # Normalize the point cloud: translation and scaling
        xyz_slc = slice(self.syntax_gt['xyz'][0], self.syntax_gt['xyz'][1] + 1)
        data[:, xyz_slc] -= data[:, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        data[:, xyz_slc] *= data[:, self.syntax_gt['block_scale']].unsqueeze(-1)

        pnts_3d = data[:, xyz_slc]
        point_feature = self.pointwise_mlp(pnts_3d) # in this case, use the xyz coordinates as feature
        if self.ext_cw:
            cw_inp1 = scatter_max(point_feature, block_idx.long(), dim=0)[0]
            cw_inp2 = scatter_min(point_feature, block_idx.long(), dim=0)[0]
            cw_inp3 = scatter_mean(point_feature, block_idx.long(), dim=0)
            cw_inp = torch.cat([cw_inp1, cw_inp2, cw_inp3], dim=1)
        else:
            cw_inp = scatter_max(point_feature, block_idx.long(), dim=0)[0]
        block_feature = self.fc(cw_inp)
        mask = data[:, self.syntax_gt['block_start']] > 0

        # Return the codeword with the meta data
        out = torch.zeros(torch.sum(mask), self.syntax_cw['__len__'], device=device)
        out[:, self.syntax_cw['cw'][0] : self.syntax_cw['cw'][1] + 1] = block_feature
        out[:, self.syntax_cw['block_pntcnt']] = data[mask, self.syntax_gt['block_pntcnt']]
        out[:, self.syntax_cw['block_center'][0] : self.syntax_cw['block_center'][1] + 1] = data[mask, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        out[:, self.syntax_cw['block_scale']] = data[mask, self.syntax_gt['block_scale']]
        out[:, self.syntax_cw['pc_start']] = pc_start
        return out