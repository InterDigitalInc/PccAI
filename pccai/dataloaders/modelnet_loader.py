# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A ModelNet data loader

import os
import os.path
import numpy as np
import pickle

import torch.utils.data as data
from torch_geometric.transforms.sample_points import SamplePoints
from  torch_geometric.datasets.modelnet import ModelNet
from pccai.utils.convert_octree import OctreeOrganizer
import pccai.utils.logger as logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path_default=os.path.abspath(os.path.join(BASE_DIR, '../../datasets/modelnet/')) # the default dataset path


def gen_rotate():
    rot = np.eye(3, dtype='float32')
    rot[0,0] *= np.random.randint(0,2) * 2 - 1
    rot = np.dot(rot, np.linalg.qr(np.random.randn(3, 3))[0])
    return rot


class ModelNetBase(data.Dataset):
    """A base ModelNet data loader."""

    def __init__(self, data_config, sele_config, **kwargs):
        if 'coord_min' in data_config or 'coord_max' in data_config:
            self.coord_minmax = [data_config.get('coord_min', 0), data_config.get('coord_max', 1023)]
        else:
            self.coord_minmax = None
        self.centralize = data_config.get('centralize', True)
        self.voxelize = data_config.get('voxelize', False)
        self.sparse_collate = data_config.get('sparse_collate', False)
        self.augmentation = data_config[sele_config].get('augmentation', False)
        self.split = data_config[sele_config]['split'].lower()
        self.num_points = data_config['num_points']
        sampler = SamplePoints(num=self.num_points, remove_faces=True, include_normals=False)
        self.point_cloud_dataset = ModelNet(root=dataset_path_default, name='40', 
            train=True if self.split == 'train' else False, transform=sampler)


    def __len__(self):
        return len(self.point_cloud_dataset)


    def pc_preprocess(self, pc):
        """Perform different types of pre-processings to the ModelNet point clouds."""
    
        if self.centralize:
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
        
        if self.augmentation: # random rotation
            pc = np.dot(pc, gen_rotate())
    
        if self.coord_minmax is not None:
            pc_min, pc_max = np.min(pc), np.max(pc)
            pc = (pc - pc_min) / (pc_max - pc_min) * (self.coord_minmax[1] - self.coord_minmax[0]) + self.coord_minmax[0]
        
            if self.voxelize:
                pc = np.unique(np.round(pc).astype('int32'), axis=0)
                # This is to facilitate the sparse tensor construction with Minkowski Engine
                if self.sparse_collate:
                    pc = np.hstack((np.zeros((pc.shape[0], 1), dtype='int32'), pc))
                    # pc = np.vstack((pc, np.ones((self.num_points - pc.shape[0], 4), dtype='int32') * -1))
                    pc[0][0] = 1
                return pc
        else: # if do not specify minmax, normalize the point cloud within a unit ball
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m # scaling
        return pc.astype('float32')


class ModelNetSimple(ModelNetBase):
    """A simple ModelNet data loader where point clouds are directly represented as 3D points."""

    def __init__(self, data_config, sele_config, **kwargs):
        super().__init__(data_config, sele_config)

        # Use_cache specifies the pickle file to be read/written down, "" means no caching mechanism is used
        self.use_cache = data_config.get('use_cache', '')

        # By using the cache file, the data is no longer generated on the fly but the loading becomes much faster
        if self.use_cache != '':
            cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../datasets/', self.use_cache)
            if os.path.exists(cache_file): # the cache file already exist
                logger.log.info("Loading pre-processed ModelNet40 cache file...")
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            else: # the cache file is not there yet
                self.cache = []
                logger.log.info("Sampling point clouds from raw ModelNet40 data...")
                for i in range(len(self.point_cloud_dataset)):
                    # Be careful that here the data type is converted as uint8 to save space
                    self.cache.append(self.pc_preprocess(self.point_cloud_dataset[i].pos.numpy()).astype(np.uint8))
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            logger.log.info("ModelNet40 data loaded...\n")

    def __getitem__(self, index):
        if self.use_cache:
            return self.cache[index].astype(np.int32) # data type convert back to int32
        else:
            return self.pc_preprocess(self.point_cloud_dataset[index].pos.numpy())


class ModelNetOctree(ModelNetBase):
    """ModelNet data loader with uniform sampling and octree partitioning."""

    def __init__(self, data_config, sele_config, **kwargs):

        data_config['voxelize'] = True
        data_config['sparse_collate'] = False
        super().__init__(data_config, sele_config)

        self.rw_octree = data_config.get('rw_octree', False)
        if self.rw_octree:
            self.rw_partition_scheme = data_config.get('rw_partition_scheme', 'default')
        self.octree_cache_folder = 'octree_cache'

        # Create an octree formatter to organize octrees into arrays
        self.octree_organizer = OctreeOrganizer(
            data_config['octree_cfg'],
            data_config[sele_config].get('max_num_points', data_config['num_points']),
            kwargs['syntax'].syntax_gt,
            self.rw_octree,
            data_config[sele_config].get('shuffle_blocks', False),
        )

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):

        while True:
            if self.rw_octree:
                file_name = os.path.join(dataset_path_default, self.octree_cache_folder, self.rw_partition_scheme, str(index)) + '.pkl'
            else: file_name = None

            # perform octree partitioning and organize the data
            pc = self.pc_preprocess(self.point_cloud_dataset[index].pos.numpy())
            pc_formatted, _, _, _, all_skip = self.octree_organizer.organize_data(pc, file_name=file_name)
            if all_skip:
                index += 1
                if index >= len(self.point_cloud_dataset): index = 0
            else: break

        return pc_formatted
