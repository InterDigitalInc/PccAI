# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A generic point cloud dataset wrapper

from torch.utils.data import DataLoader
from pccai.dataloaders.shapenet_part_loader import ShapeNetPart
from pccai.dataloaders.modelnet_loader import ModelNetSimple, ModelNetOctree
from pccai.dataloaders.lidar_loader import LidarSimple, LidarSpherical, LidarOctree
import torch
import numpy as np


# https://github.com/pytorch/pytorch/issues/5059
# Fix numpy random seed issue with multi worker DataLoader
# Multi worker based on process forking duplicates the same numpy random seed across all workers
# Note that this issue is absent with pytorch random operations
def wif(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def get_point_cloud_dataset(dataset_name):
    """List all the data sets in this function for class retrival."""

    if dataset_name.lower() == 'shapenet_part':
        dataset_class = ShapeNetPart
    elif dataset_name.lower() == 'modelnet_simple':
        dataset_class = ModelNetSimple
    elif dataset_name.lower() == 'modelnet_octree':
        dataset_class = ModelNetOctree
    elif dataset_name.lower().find('simple') >= 0:
        dataset_class = LidarSimple
    elif dataset_name.lower().find('spherical') >= 0:
        dataset_class = LidarSpherical
    elif dataset_name.lower().find('octree') >= 0:
        dataset_class = LidarOctree
    else:
        dataset_class = None
    return dataset_class


def sparse_collate(list_data):
    """A collate function tailored for generating sparse voxels of MinkowskiEngine."""

    list_data = np.vstack(list_data)
    list_data = torch.from_numpy(list_data)
    return list_data


def point_cloud_dataloader(data_config, syntax=None, ddp=False):
    """A wrapper for point cloud datasets."""

    point_cloud_dataset = get_point_cloud_dataset(data_config[0]['dataset'])(data_config[0], data_config[1], syntax=syntax)
    collate_fn = sparse_collate if data_config[0].get('sparse_collate', False) else None
    dl_conf = data_config[0][data_config[1]]

    if ddp: # for distributed data parallel
        sampler = torch.utils.data.distributed.DistributedSampler(point_cloud_dataset, shuffle=dl_conf['shuffle'])
        point_cloud_dataloader = DataLoader(point_cloud_dataset, batch_size=int(dl_conf['batch_size'] / torch.cuda.device_count()),
            num_workers=int(dl_conf['num_workers'] / torch.cuda.device_count()), persistent_workers=True if dl_conf['num_workers'] > 0 else False,
            worker_init_fn=wif, sampler=sampler, pin_memory=False, drop_last=False, collate_fn=collate_fn)
    else:
        point_cloud_dataloader = DataLoader(point_cloud_dataset, batch_size=dl_conf['batch_size'], shuffle=dl_conf['shuffle'],
            num_workers=dl_conf['num_workers'], persistent_workers=True if dl_conf['num_workers'] > 0 else False,
            worker_init_fn=wif, pin_memory=False, drop_last=False, collate_fn=collate_fn)
    return point_cloud_dataset, point_cloud_dataloader