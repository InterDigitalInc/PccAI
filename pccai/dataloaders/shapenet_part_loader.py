# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A ShapeNet-Part data loader

import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import pccai.utils.logger as logger
import multiprocessing
from tqdm import tqdm
from functools import partial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path_default=os.path.abspath(os.path.join(BASE_DIR, '../../datasets/shapenet_part/')) # the default dataset path


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_pc(index, datapath, classes, normalize):
    fn = datapath[index]
    cls = classes[datapath[index][0]]
    point_set = np.loadtxt(fn[1]).astype(np.float32)
    if normalize:
        point_set = pc_normalize(point_set)
    seg = np.loadtxt(fn[2]).astype(np.int64) - 1
    foldername = fn[3]
    filename = fn[4]
    return (point_set, seg, cls, foldername, filename)


class ShapeNetPart(data.Dataset):
    """A ShapeNet part dataset class."""

    def __init__(self, data_config, sele_config, **kwargs):
        # Common options of the dataset
        dataset_path = data_config.get('dataset_path', dataset_path_default)
        dataset_path = os.path.join(dataset_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
        # Allow override of num_points in specific modes
        # null (YAML) / None (Python) means no sampling
        num_points = data_config[sele_config].get('num_points', data_config.get('num_points', 2500))
        classification = data_config.get('classification', False)
        normalize = data_config.get('normalize', True)

        # Options under a specific configuration
        class_choice = data_config[sele_config].get('class_choice', None)
        split = data_config[sele_config].get('split', 'train')
        augmentation = data_config[sele_config].get('augmentation', False)
         # Should perform augmentation in __getitem__() if needed
        self.num_points = num_points
        self.catfile = os.path.join(dataset_path, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
            logger.log.info(self.cat)
        self.meta = {}
        with open(os.path.join(dataset_path, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(dataset_path, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(dataset_path, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(dataset_path, self.cat[item], 'points')
            dir_seg = os.path.join(dataset_path, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                logger.log.info('Unknown split: %s. Exiting..' % (split))
                exit(0)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),self.cat[item], token))            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        logger.log.info(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

        load_pc_part = partial(load_pc, datapath=self.datapath, classes=self.classes, normalize=self.normalize)

        self.cache = np.empty(len(self.datapath), dtype=object)
        if not data_config.get('lazy_loading', False):
            # Precaching
            with multiprocessing.Pool() as p:
                self.cache = np.array(list(tqdm(p.imap(load_pc_part, np.arange(len(self.datapath)), 32), total=len(self.datapath))), dtype=object)

    def __getitem__(self, index):
        value = self.cache[index]
        if value is None:
            value = self.cache[index] = load_pc(index, self.datapath, self.classes, self.normalize)
        point_set, seg, cls, foldername, filename = value

        if self.num_points is not None:
            choice = np.random.choice(len(seg), self.num_points, replace=True)
            # resample
            point_set = point_set[choice, :]
        
        # To Pytorch
        point_set = torch.from_numpy(point_set)
        return point_set
        

    def __len__(self):
        return len(self.datapath)
