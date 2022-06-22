# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A multi-modal data loader for LiDAR datasets.

import os
import numpy as np
from torch.utils import data

from pccai.utils.convert_image import pc2img
from pccai.utils.convert_octree import OctreeOrganizer
from pccai.dataloaders.lidar_base_loader import FordBase, KITTIBase, QnxadasBase


def get_base_lidar_dataset(data_config, sele_config):
    if data_config['dataset'].lower().find('ford') >= 0:
        loader_class = FordBase
    elif data_config['dataset'].lower().find('kitti') >= 0:
        loader_class = KITTIBase
    elif data_config['dataset'].lower().find('qnxadas') >= 0:
        loader_class = QnxadasBase
    else:
        loader_class = None
    return loader_class(data_config, sele_config)


class LidarSimple(data.Dataset):
    """A simple LiDAR dataset which returns a specified number of 3D points in each point cloud."""

    def __init__(self, data_config, sele_config, **kwargs):

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)
        self.num_points = data_config.get('num_points', 150000) # about 150000 points per point cloud
        self.seed = data_config.get('seed', None)
        self.sparse_collate = data_config.get('sparse_collate', False)
        self.voxelize = data_config.get('voxelize', False)

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def __getitem__(self, index):
        pc = self.point_cloud_dataset[index]['pc'] # take out the point cloud coordinates only
        np.random.seed(self.seed)
        if self.voxelize:
            pc = np.round(pc[:self.num_points, :]).astype('int32') # always <= num_points
            # This is to facilitate the sparse tensor construction with Minkowski Engine
            if self.sparse_collate:
                pc = np.hstack((np.zeros((pc.shape[0], 1), dtype='int32'), pc))
                # pc = np.vstack((pc, np.ones((self.num_points - pc.shape[0], 4), dtype='int32') * -1))
                pc[0][0] = 1
            return pc
        else:
            choice = np.random.choice(pc.shape[0], self.num_points, replace=True) # always == num_points
            return pc[choice, :].astype(dtype=np.float32)

class LidarSpherical(data.Dataset):
    """Converts the original Cartesian coordinate to spherical coordinate then represent as 2D images."""

    def __init__(self, data_config, sele_config, **kwargs):

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)
        self.width = data_config['spherical_cfg'].get('width', 1024) # grab all the options about speherical projection
        self.height = data_config['spherical_cfg'].get('height', 128)
        self.v_fov = data_config['spherical_cfg'].get('v_fov', [-28, 3.0])
        self.h_fov = data_config['spherical_cfg'].get('h_fov', [-180, 180])
        self.origin_shift = data_config['spherical_cfg'].get('origin_shift', [0, 0, 0])
        self.v_fov, self.h_fov = np.array(self.v_fov) / 180 * np.pi, np.array(self.h_fov) / 180 * np.pi
        self.num_points = self.width * self.height
        self.inf = 1e6

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]['pc'] # take out the point cloud coordinates only
        data[:, 0] += self.origin_shift[0]
        data[:, 1] += self.origin_shift[1]
        data[:, 2] += self.origin_shift[2]
        data_img = pc2img(self.h_fov, self.v_fov, self.width, self.height, self.inf, data)        

        return data_img


class LidarOctree(data.Dataset):
    """Converts an original point cloud into an octree."""

    def __init__(self, data_config, sele_config, **kwargs):

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)
        self.rw_octree = data_config.get('rw_octree', False)
        if self.rw_octree:
            self.rw_partition_scheme = data_config.get('rw_partition_scheme', 'default')
        self.octree_cache_folder = 'octree_cache'

        # Create an octree formatter to organize octrees into arrays
        self.octree_organizer = OctreeOrganizer(
            data_config['octree_cfg'],
            data_config[sele_config].get('max_num_points', 150000),
            kwargs['syntax'].syntax_gt,
            self.rw_octree,
            data_config[sele_config].get('shuffle_blocks', False),
        )

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):

        if self.rw_octree:
            file_name = os.path.relpath(self.point_cloud_dataset.get_pc_idx(index), self.point_cloud_dataset.dataset_path)
            file_name = os.path.join(self.point_cloud_dataset.dataset_path, self.octree_cache_folder, self.rw_partition_scheme, file_name)
            file_name = os.path.splitext(file_name)[0] + '.pkl'
        else: file_name = None

        pc = self.point_cloud_dataset[index]['pc']
        # perform octree partitioning and organize the data
        pc_formatted, _, _, _, _ = self.octree_organizer.organize_data(pc, file_name=file_name)

        return pc_formatted
