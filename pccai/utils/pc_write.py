# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import os
import numpy as np
import open3d as o3d
import pccai.utils.logger as logger


def pc_write_o3d(pc, file_name, coloring=None, normals=None):
    """Basic writing tool for point clouds."""

    pc = pt_to_np(pc)
    pc_o3d = o3d.geometry.PointCloud()
    try:
        pc_o3d.points = o3d.utility.Vector3dVector(pc)
        if coloring is not None:
            pc_o3d.colors = o3d.utility.Vector3dVector(coloring)
        if normals is not None:
            pc_o3d.normals = o3d.utility.Vector3dVector(normals)
    except RuntimeError as e:
        logger.log.info(pc, coloring, normals)
        logger.log.info(type(pc), type(coloring), type(normals))
        raise e
    o3d.io.write_point_cloud(file_name, pc_o3d)


def pt_to_np(tensor):
    """Convert PyTorch tensor to NumPy array."""

    return tensor.contiguous().cpu().detach().numpy()
