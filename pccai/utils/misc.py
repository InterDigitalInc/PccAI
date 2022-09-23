# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import numpy as np
import pccai.utils.logger as logger
from plyfile import PlyData, PlyElement


def pc_write(pc, file_name):
    pc_np = pc.T.cpu().numpy()
    vertex = list(zip(pc_np[0], pc_np[1], pc_np[2]))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    elements = PlyElement.describe(vertex, "vertex")
    PlyData([elements]).write(file_name)
    return


def pc_read(filename):
    ply_raw = PlyData.read(filename)['vertex'].data
    pc = np.vstack((ply_raw['x'], ply_raw['y'], ply_raw['z'])).transpose()
    return np.ascontiguousarray(pc)


def pt_to_np(tensor):
    """Convert PyTorch tensor to NumPy array."""

    return tensor.contiguous().cpu().detach().numpy()


def load_state_dict_with_fallback(obj, dict):
    """Load a checkpoint with fall back."""

    try:
        obj.load_state_dict(dict)
    except RuntimeError as e:
        logger.log.exception(e)
        logger.log.info(f'Strict load_state_dict has failed. Attempting in non strict mode.')
        obj.load_state_dict(dict, strict=False)