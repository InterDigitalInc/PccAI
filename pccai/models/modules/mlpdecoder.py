# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# MLP Decoder

import torch
import torch.nn as nn
import numpy as np
from pccai.models.utils import PointwiseMLP


class MlpDecoder(nn.Module):
    """MLP decoder in homogeneous batching mode."""

    def __init__(self, net_config, **kwargs):
        super(MlpDecoder, self).__init__()
        self.num_points = net_config['num_points']
        dims = net_config['dims']
        self.mlp = PointwiseMLP(dims + [3 * self.num_points], doLastRelu=False) # the MLP layers

    def forward(self, cw):

        out1 = self.mlp(cw) # BatchSize X PointNum X 3
        return out1.view(cw.shape[0], self.num_points, -1)


class MlpDecoderHetero(nn.Module):
    """MLP decoder for heterogeneous batching."""

    def __init__(self, net_config, **kwargs):
        super(MlpDecoderHetero, self).__init__()
        self.num_points = net_config['num_points']
        dims = net_config['dims']
        self.mlp = PointwiseMLP(dims + [3 * self.num_points], doLastRelu=False) # the MLP layers

        # Grab the syntax
        self.syntax_cw = kwargs['syntax'].syntax_cw
        self.syntax_rec = kwargs['syntax'].syntax_rec

    def forward(self, cw):
        device = cw.device
        pc_block = self.mlp(cw[:, self.syntax_cw['cw'][0] : self.syntax_cw['cw'][1] + 1]) # apply MLP layers directly
        pc_block = pc_block.view(cw.shape[0] * self.num_points, -1)

        block_npts = torch.ones(cw.shape[0], dtype=torch.long, device=device) * self.num_points
        # For each point, indice the index of its codeword/block
        cw_idx = torch.arange(block_npts.shape[0], device=device).repeat_interleave(block_npts)
        # Mark a point with 1 if it is the first point of a block
        block_start = torch.cat((torch.ones(1, device=device), cw_idx[1:] - cw_idx[:-1])).float()

        # Denormalize the point cloud
        center = cw[:, self.syntax_cw['block_center'][0]: self.syntax_cw['block_center'][1] + 1].repeat_interleave(block_npts, 0)
        scale = cw[:, self.syntax_cw['block_scale']: self.syntax_cw['block_scale'] + 1].repeat_interleave(block_npts, 0)

        # From pc_start in cw (blocks), build pc_start for points
        pc_start = torch.zeros(cw.shape[0], device=device).repeat_interleave(block_npts)
        # Starting point index for each block
        block_idx = torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(block_npts, 0)[:-1]), 0)
        # Mark a point as one if it is the first of its point cloud
        # We have this binary marker for each block of the point cloud (1 if first block, 0 otherwise)
        # We mark the first point of all blocks with the marker of their block
        pc_start[block_idx] = cw[:, self.syntax_cw['pc_start']: self.syntax_cw['pc_start'] + 1].squeeze(-1)

        # Denormalization: scaling and translation
        pc_block = pc_block / scale # scaling
        pc_block = pc_block + center # translation

        # Assemble the output
        out = torch.zeros(pc_block.shape[0], self.syntax_rec['__len__']).cuda()
        out[:, self.syntax_rec['xyz'][0] : self.syntax_rec['xyz'][1] + 1] = pc_block
        out[:, self.syntax_rec['block_start']] = block_start
        out[:, self.syntax_rec['block_center'][0] : self.syntax_rec['block_center'][1] + 1] = center
        out[:, self.syntax_rec['block_scale']] = scale[:, 0]
        out[:, self.syntax_rec['pc_start']] = pc_start
        return out


    def prepare_meta_data(self, binstrs, block_pntcnt, octree_organizer):
        """Convert the binary strings of an octree to a set of scales and centers of the leaf nodes.
        Next, arranges them as the meta data array according to the syntax for decoding.
        """

        leaf_blocks = octree_organizer.departition_octree(binstrs, block_pntcnt) # departition the octree strings to blocks
        meta_data = np.zeros((len(leaf_blocks), self.syntax_cw['__len__'] - self.syntax_cw['__meta_idx__']), dtype=np.float32)
        cur = 0

        # Assemble the meta data
        meta_data[0, self.syntax_cw['pc_start'] - self.syntax_cw['__meta_idx__']] = 1
        for idx, block in enumerate(leaf_blocks):
            if block['binstr'] >= 0: # only keep the blocks with transform mode
                center, scale = octree_organizer.get_normalizer(block['bbox_min'], block['bbox_max'])
                meta_data[cur, self.syntax_cw['block_pntcnt'] - self.syntax_cw['__meta_idx__']] = block_pntcnt[idx]
                meta_data[cur, self.syntax_cw['block_scale'] - self.syntax_cw['__meta_idx__']] = scale
                meta_data[cur, self.syntax_cw['block_center'][0] - self.syntax_cw['__meta_idx__'] : 
                    self.syntax_cw['block_center'][1] - self.syntax_cw['__meta_idx__'] + 1] = center
                cur += 1

        # Only returns the useful part
        return torch.as_tensor(meta_data[:cur, :], device=torch.device('cuda')).unsqueeze(-1).unsqueeze(-1)