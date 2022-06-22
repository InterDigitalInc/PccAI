# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# This is an example PCC Codec based on octree partitioning, then each block is digested and compressed individually

import torch
import gzip
import numpy as np
import time

from pccai.utils.convert_octree import OctreeOrganizer
from pccai.codecs.pcc_codec import PccCodecBase


class OctreePartitionCodec(PccCodecBase):
    """An example PCC Codec based on octree partitioning and blockwise processing."""

    def __init__(self, codec_config, pccnet, bit_depth, syntax):
        super().__init__(codec_config, pccnet, syntax)
        self.pc_organizer = OctreeOrganizer(
            codec_config['octree_cfg'],
            codec_config['max_num_points'],
            syntax.syntax_gt
        )
        self.cw_shape = torch.Size([1, 1])


    def compress(self, points, tag):
        """Compress all the blocks of a point cloud then write the bitstream to a file."""
    
        start = time.monotonic()
        file_name = tag + '.bin'
        points = (points + np.array(self.translate)) * self.scale
        points, _, octree_strs, block_pntcnt, _ = self.pc_organizer.organize_data(points)
        points = torch.from_numpy(points).cuda()
        compress_out, _ = self.pccnet.compress(points.unsqueeze(0)) # perform compression
        pc_strs = compress_out['strings'][0]
        end = time.monotonic()

        # Write down the point cloud on disk
        with gzip.open(file_name, 'wb') as f:
            ret = save_pc_stream(pc_strs, octree_strs, block_pntcnt)
            f.write(ret)
        
        # Return other statistics through this dictionary
        stat_dict = {
            'enc_time': round(end - start, 3),
        }

        return [file_name], stat_dict
    
    
    def decompress(self, file_name):
        """Decompress all the blocks of a point cloud from a file."""
    
        with gzip.open(file_name[0], 'rb') as f:
            pc_strs, octree_strs, block_pntcnt = load_pc_stream(f)

        start = time.monotonic()
        meta_data = self.pccnet.decoder.prepare_meta_data(octree_strs, block_pntcnt, self.pc_organizer)
    
        # Decompress the point cloud
        pc_rec, _ = self.pccnet.decompress([pc_strs], self.cw_shape, meta_data)
        pc_rec = (pc_rec / self.scale - torch.tensor(self.translate, device=pc_rec.device)).long() # denormalize
        end = time.monotonic()

        # Return other statistics through this dictionary
        stat_dict = {
            'dec_time': round(end - start, 3),
        }
    
        return pc_rec, stat_dict


def save_pc_stream(pc_strs, octree_strs, block_pntcnt):
    """Save an octree-partitioned point cloud and its partitioning information as an unified bitstream."""

    n_octree_str_b = array_to_bytes([len(octree_strs)], np.uint16) # number of nodes in the octree
    n_blocks_b = array_to_bytes([len(block_pntcnt)], np.uint16) # number of blocks in total
    n_trans_block_b = array_to_bytes([len(pc_strs)], np.uint16) # number of blocks that are coded with transformed mode
    octree_strs_b = array_to_bytes(octree_strs, np.uint8) # bit stream of the octree 
    pntcnt_b = array_to_bytes(block_pntcnt, np.uint16) # bit stream of the point count in each block
    out_stream = n_octree_str_b + n_blocks_b + n_trans_block_b + octree_strs_b + pntcnt_b

    # Work on each block of the point cloud
    for strings in pc_strs:
        n_bytes_b = array_to_bytes([len(strings)], np.uint16) # number of bytes spent in the current block
        out_stream += n_bytes_b + strings
    return out_stream


def load_pc_stream(f):
    """Load an octree-partitioned point cloud unified bitstream."""

    n_octree_str = load_buffer(f, 1, np.uint16)[0]
    n_blocks = load_buffer(f, 1, np.uint16)[0]
    n_trans_block = load_buffer(f, 1, np.uint16)[0]
    octree_strs = load_buffer(f, n_octree_str, np.uint8)
    block_pntcnt = load_buffer(f, n_blocks, np.uint16)

    pc_strs = []
    for _ in range(n_trans_block):
        n_bytes = load_buffer(f, 1, np.uint16)[0]
        string = f.read(int(n_bytes))
        pc_strs.append(string)
    file_end = f.read()
    assert file_end == b'', f'File not read completely file_end {file_end}'

    return pc_strs, octree_strs, block_pntcnt

    
def array_to_bytes(x, dtype):
    x = np.array(x, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        type_info = np.finfo(dtype)
    else:
        type_info = np.iinfo(dtype)
    assert np.all(x <= type_info.max), f'Overflow {x} {type_info}'
    assert np.all(type_info.min <= x), f'Underflow {x} {type_info}'
    return x.tobytes()


def load_buffer(file, cnt, dtype):
    return np.frombuffer(file.read(int(np.dtype(dtype).itemsize * cnt)), dtype=dtype)