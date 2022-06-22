# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Compute Chamfer Distance loss for raw point clouds (homogeneous or heterogeneous)

import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase


def nndistance_simple(rec, data):
    """
    A simple nearest neighbor search, not very efficient, just for reference
    """
    rec_sq = torch.sum(rec * rec, dim=2, keepdim=True)  # (B,N,1)
    data_sq = torch.sum(data * data, dim=2, keepdim=True)  # (B,M,1)
    cross = torch.matmul(data, rec.permute(0, 2, 1))         # (B,M,N)
    dist = data_sq - 2 * cross + rec_sq.permute(0, 2, 1)       # (B,M,N)
    data_dist, data_idx = torch.min(dist, dim=2)
    rec_dist, rec_idx = torch.min(dist, dim=1)
    return data_dist, rec_dist, data_idx, rec_idx


try:
    # If you want to use the efficient NN search for computing CD loss, compiled the nndistance()
    # function under the third_party folder according to instructions in Readme.md
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))
    from modules.nnd import NNDModule
    nndistance = NNDModule()
except ModuleNotFoundError:
    # Without the compiled nndistance(), by default the nearest neighbor will be done using pytorch-geometric
    nndistance = nndistance_simple


class ChamferDistCanonical(PccLossBase):
    """Chamfer distance loss for both homogeneous and heterogeneous batching."""

    def __init__(self, loss_args, syntax):
        super().__init__(loss_args, syntax)
        self.xyz_loss_type = loss_args['xyz_loss_type']
        self.xyz_subset_weight = loss_args.get('xyz_subset_weight', 1.0) # weight of the subset distance
        self.inf =1e12
        
        # Syntax of the ground-truth and the reconstruction
        self.syntax_gt = syntax.syntax_gt
        self.syntax_rec = syntax.syntax_rec


    def dist_hetero(self, data, rec, batch_size):
        """Compute birectional distances between two point clouds in heterogeneous mode."""

        device = data.device
        batch_size = data.shape[0]

        # Get the number of points for each point cloud in the reconstruction
        pnt_cnt_batch = torch.ones(batch_size + 1, dtype=torch.int32, device=device) * rec.shape[0]
        pnt_cnt_batch[0 : -1] = torch.arange(rec.shape[0], device=device)[rec[:, self.syntax_rec['pc_start']] > 0]
        pnt_cnt_batch = pnt_cnt_batch[1:] - pnt_cnt_batch[0 : -1]
        max_pnts_batch = max(pnt_cnt_batch) # get the maximum number of points among all point clouds
        avail_idx = torch.cat([torch.arange(n, dtype=torch.long, device=device) + idx * max_pnts_batch
            for idx, n in enumerate(pnt_cnt_batch)]) # obtain the indices of the available points

        rec_homo = torch.ones((max_pnts_batch * batch_size, 3), device=device) * self.inf # set the pading to inf
        rec_homo[avail_idx, :] = rec[:, self.syntax_rec['xyz'][0] : self.syntax_rec['xyz'][1] + 1]
        rec_homo = rec_homo.view(batch_size, -1, 3) # build a homogeneous 3D tensor holding the reconstrcutions

        # Build a homogeneous 3D tensor holding the ground-truths
        data_homo = data[:, :, self.syntax_gt['xyz'][0] : self.syntax_gt['xyz'][1] + 1].clone()
        data_homo = data_homo.view(-1, 3)
        data = data.view(-1, self.syntax_gt['__len__'])
        data_homo[data[:, self.syntax_gt['block_pntcnt']] <= 0, :] = self.inf # set the padding to inf
        data_homo = data_homo.view(batch_size, -1, 3)

        # Compute the nearest neighbor distances, then retrieve the available distance values
        data_dist, rec_dist, _, _ = nndistance(data_homo.contiguous(), rec_homo.contiguous())
        data_dist = data_dist.view(-1)[data[:, self.syntax_gt['block_pntcnt']] > 0]
        rec_dist = rec_dist.view(-1)[avail_idx]
        rep_times = data_homo.shape[1]
        return data_dist, rec_dist, rep_times


    def xyz_loss(self, loss_out, net_in, net_out):
        """Chamfer distance computation using nndistance()."""

        rec = net_out['x_hat']
        batch_size = net_in.shape[0]
        loss = 0
        if self.hetero:
            data_dist, rec_dist, rep_times = self.dist_hetero(net_in, rec, batch_size) # compute the Chamfer distance values
            if self.xyz_loss_type.find('l1') >= 0: # compute square root if l1-norm is used
                data_dist, rec_dist = data_dist ** 0.5, rec_dist ** 0.5
            rec_dist = rec_dist * self.xyz_subset_weight # weight the subset distance
            if self.xyz_loss_type.find('max') >= 0: # use max function for aggregation
                net_in = net_in.view(-1, self.syntax_gt['__len__'])
                memb_data = torch.arange(0, batch_size, dtype=torch.long, device=net_in.device).repeat_interleave(rep_times)
                memb_data = memb_data[(net_in[:, self.syntax_gt['block_pntcnt']] > 0).view(-1)] # membership of each point in the batch of gt_data
                memb_rec = torch.cumsum(rec[:, self.syntax_rec['pc_start']], dim=0).long() - 1 # membership of each point in the batch of rec
                losses = torch.stack([torch.max(torch.mean(data_dist[memb_data==idx]), torch.mean(rec_dist[memb_rec==idx]))
                                      for idx in range(batch_size)])
                loss = torch.mean(losses)
            else:
                loss = torch.mean(data_dist + rec_dist)
        else:
            net_in = net_in.contiguous()
            rec = rec.contiguous()
            data_dist, rec_dist, _, _ = nndistance(net_in, rec)
            if self.xyz_loss_type.find('l1') >= 0: # compute square root if l1-norm is used
                data_dist, rec_dist = data_dist ** 0.5, rec_dist ** 0.5
            rec_dist = rec_dist * self.xyz_subset_weight # weight the subset distance
            data_dist, rec_dist = torch.mean(data_dist, 1), torch.mean(rec_dist, 1)
            if self.xyz_loss_type.find('max') >= 0: # use max function for aggregation
                loss = torch.mean(torch.max(data_dist, rec_dist))
            else: loss = torch.mean(data_dist + rec_dist)

        loss_out['xyz_loss'] = loss.unsqueeze(0) # write the 'xyz_loss' as return


    def loss(self, net_in, net_out):
        """Overall R-D loss computation."""

        loss_out = {}

        # Rate loss
        if 'likelihoods' in net_out and len(net_out['likelihoods']) > 0:
            count = torch.sum(net_in[:, :, self.syntax_gt['block_pntcnt']] > 0) if self.hetero else net_in.shape[0] * net_in.shape[1]
            self.bpp_loss(loss_out, net_out['likelihoods'], count)
        else:
            loss_out['bpp_loss'] = torch.zeros((1,))
            if net_out['x_hat'].is_cuda:
                loss_out['bpp_loss'] = loss_out['bpp_loss'].cuda()
        
        # Distortion loss
        self.xyz_loss(loss_out, net_in, net_out)

        # R-D loss = alpha * D + beta * R
        loss_out["loss"] = self.alpha * loss_out['xyz_loss'] +  self.beta * loss_out["bpp_loss"] # R-D loss

        return loss_out
