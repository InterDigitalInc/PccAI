# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Base LiDAR data sets, includeing Ford, KITTI and Qnxadas

import os
import numpy as np
from torch.utils import data
from pccai.utils.misc import pc_read

found_quantize = False


def absoluteFilePaths(directory):
   for dirpath, _, file_names in os.walk(directory):
       for f in file_names:
           yield os.path.abspath(os.path.join(dirpath, f))


class FordBase(data.Dataset):
    """A base Ford dataset."""

    def __init__(self, data_config, sele_config, **kwargs):

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Common options of the dataset
        self.return_intensity = data_config.get('return_intensity', False)
        self.dataset_path = data_config.get('dataset_path', '../../datasets/ford/') # the default dataset path
        self.dataset_path = os.path.abspath(os.path.join(base_dir, self.dataset_path))
        self.translate = data_config.get('translate', [0, 0, 0])
        self.scale = data_config.get('scale', 1)
        self.point_max = data_config.get('point_max', -1)

        # Options under a specific configuration
        self.split = data_config[sele_config]['split']
        splitting = data_config['splitting'][self.split]

        self.im_idx = []
        for i_folder in splitting:
            folder_path = os.path.join(self.dataset_path, 'Ford_' + str(i_folder).zfill(2) + '_q_1mm')
            assert os.path.exists(folder_path), f'{folder_path} does not exist'
            self.im_idx += absoluteFilePaths(folder_path)
        self.im_idx.sort()


    def __len__(self):
        """Returns the total number of samples"""
        return len(self.im_idx)


    def __getitem__(self, index):
        
        pc = (pc_read(self.im_idx[index]) + np.array(self.translate)) * self.scale
        if self.point_max > 0 and pc.shape[0] > self.point_max:
                pc = pc[:self.point_max, :]
        return {'pc': pc, 'ref': None}


    def get_pc_idx(self, index):
        return self.im_idx[index]


class QnxadasBase(data.Dataset):
    """A base Qnxadas dataset."""

    def __init__(self, data_config, sele_config, **kwargs):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path_default = os.path.abspath(os.path.join(base_dir, '../../datasets/qnxadas/')) # the default dataset path

        # Common options of the dataset
        self.return_intensity = data_config.get('return_intensity', False)
        dataset_path = data_config.get('dataset_path', dataset_path_default)
        self.translate = data_config.get('translate', [0, 0, 0])
        self.scale = data_config.get('scale', 1)

        # Options under a specific configuration
        self.split = data_config[sele_config]['split']
        splitting = data_config['splitting'][self.split]

        self.im_idx = []
        for i_folder in splitting:
            self.im_idx += absoluteFilePaths(os.path.join(dataset_path, i_folder))
        self.im_idx.sort()


    def __len__(self):
        """Returns the total number of samples"""
        return len(self.im_idx) // 2


    def __getitem__(self, index):
        pc = (pc_read(self.im_idx[2 * index + 1]) + np.array(self.translate)) * self.scale
        return {'pc': pc, 'ref': None}


    def get_pc_idx(self, index):
        return self.im_idx[2 * index + 1]


class KITTIBase(data.Dataset):
    """A base SemanticKITTI dataset."""

    def __init__(self, data_config, sele_config, **kwargs):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.abspath(os.path.join(base_dir, '../../datasets/kitti/')) # the default dataset path
        
        # Other specific options
        self.translate = data_config.get('translate', [0, 0, 0])
        self.scale = data_config.get('scale', 1)
        self.quantize_resolution = data_config.get('quantize_resolution', None) if found_quantize else None
        self.split = data_config[sele_config]['split']
        splitting = data_config['splitting'][self.split]
        
        self.im_idx = []
        for i_folder in splitting:
            self.im_idx += absoluteFilePaths('/'.join([dataset_path, str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()


    def __len__(self):
        """Returns the total number of samples"""
        return len(self.im_idx)


    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.quantize_resolution is not None:
            pc = quantize_resolution(raw_data[:, :3], self.quantize_resolution)
        else:
            pc = (raw_data[:, :3] + np.array(self.translate)) * self.scale
        return {'pc': pc}


    def get_pc_idx(self, index):
        return self.im_idx[index]
