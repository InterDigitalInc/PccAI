# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Octree partitioning and departitioning with breadth-first search

import os
import pickle
import numpy as np
# from numba import njit


# @njit
def compute_new_bbox(idx, bbox_min, bbox_max):
    """Compute global block bounding box given an index."""

    midpoint = (bbox_min + bbox_max) / 2
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()
    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max


# @njit
def _analyze_octant(points, bbox_min, bbox_max):
    """Analyze the statistics of the points in a given block."""

    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2

    locations = (points >= np.expand_dims(center, 0)).astype(np.uint8)
    locations *= np.array([[1, 2, 4]], dtype=np.uint8)
    locations = np.sum(locations, axis=1)

    location_cnt = np.zeros((8,), dtype=np.uint32)
    for idx in range(locations.shape[0]):
        loc = locations[idx]
        location_cnt[loc] += 1

    location_map = np.zeros(locations.shape[0], dtype=np.uint32)
    location_idx = np.zeros((8,), dtype=np.uint32)
    for i in range(1, location_idx.shape[0]):
        location_idx[i] = location_idx[i-1] + location_cnt[i-1]
    for idx in range(locations.shape[0]):
        loc = locations[idx]
        location_map[location_idx[loc]] = idx
        location_idx[loc] += 1

    # occupancy pattern of current node
    pattern = np.sum((location_cnt > 0).astype(np.uint32) * np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint32))
    points = points[location_map, :] # rearrange the points
    child_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]

    return points, location_cnt, pattern, child_bboxes, location_map


def analyze_octant(points, bbox_min, bbox_max, attr=None):
    points, location_cnt, pattern, child_bboxes, location_map = _analyze_octant(points, bbox_min, bbox_max)
    if attr is not None:
        attr = attr[location_map, :]
    
    return points, location_cnt, pattern, child_bboxes, attr


class OctreeConverter():
    """
    A class to store the octree paramters and perform octree partitioning.
    """

    def __init__(self, bbox_min, bbox_max, point_min, point_max, level_min, level_max):
    
        # Set the octree partitioning options
        self.bbox_min, self.bbox_max = np.asarray(bbox_min, dtype=np.float32), np.asarray(bbox_max, dtype=np.float32)
        # self.bbox_min, self.bbox_max = np.asarray(bbox_min, dtype=np.int32), np.asarray(bbox_max, dtype=np.int32)
        self.point_min, self.point_max = point_min, point_max
        self.level_min, self.level_max = level_min, level_max
        self.normalized_box_size = 2


    def leaf_test(self, point_cnt, level):
        """Determine whether a block is a leaf."""
        return (level >= self.level_max) or (point_cnt <= self.point_max and level >= self.level_min)


    def skip_test(self, point_cnt):
        """Determine whether a block should be skipped or not."""
        return point_cnt < self.point_min # True: skip; False: Transform


    def partition_octree(self, points, attr=None):
        """Octree partitioning with breadth-first search."""

        # Remove the points out of bounding box
        mask = np.ones(points.shape[0], dtype=bool)
        for i in range(3):
            mask = mask & (points[:, i] >= self.bbox_min[i]) & (points[:, i] <= self.bbox_max[i])
        points = points[mask,:]
        if attr is not None: attr = attr[mask,:]

        # initialization
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max, 'pnt_range': np.array([0, points.shape[0] - 1]), 'parent': -1, 'binstr': 0}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        # Start the splitting
        while True:
            pnt_start, pnt_end = blocks[cur]['pnt_range'][0], blocks[cur]['pnt_range'][1]
            point_cnt = pnt_end - pnt_start + 1
            if self.leaf_test(point_cnt, blocks[cur]['level']): # found a leaf node
                leaf_idx.append(cur)
                if self.skip_test(point_cnt): # Use skip transform if very few points
                    blocks[cur]['binstr'] = -1 # -1 - "skip"; 0 - "transform"
            else: # split current node
                points[pnt_start : pnt_end + 1], location_cnt, blocks[cur]['binstr'], child_bboxes, attr_tmp = \
                    analyze_octant(points[pnt_start : pnt_end + 1], blocks[cur]['bbox_min'], blocks[cur]['bbox_max'],
                    attr[pnt_start : pnt_end + 1] if attr is not None else None)
                if attr is not None: attr[pnt_start : pnt_end + 1] = attr_tmp

                # Create the child nodes            
                location_idx = np.insert(np.cumsum(location_cnt, dtype=np.uint32), 0, 0) + blocks[cur]['pnt_range'][0]
                for idx in range(8):
                    if location_cnt[idx] > 0: # creat a child node if still have points
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': child_bboxes[idx][0], 'bbox_max': child_bboxes[idx][1],
                            'pnt_range': np.array([location_idx[idx], location_idx[idx + 1] - 1], dtype=location_idx.dtype),
                            'parent': cur, 'binstr': 0}
                        blocks.append(block)
            cur += 1
            if cur >= len(blocks): break

        binstrs = np.asarray([np.max((blocks[i]['binstr'], 0)) for i in range(len(blocks))]).astype(np.uint8) # the final binary strings are always no less than 0
        return blocks, leaf_idx, points, attr, binstrs


    def departition_octree(self, binstrs, block_pntcnt):
        """Departition a given octree with breadth-first search.
        Given the binary strings and the bounding box, recover the bounding boxes and the levels of every leaf nodes.
        """

        # Initialization
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        while True:
            blocks[cur]['binstr'] = binstrs[cur]
            if blocks[cur]['binstr'] <= 0:
                leaf_idx.append(cur) # found a leaf node
                if self.skip_test(block_pntcnt[len(leaf_idx) - 1]):
                    blocks[cur]['binstr'] = -1 # marked as a skip
                else:
                    blocks[cur]['binstr'] = 0 # marked as transform
            else: # split current node
                idx = 0
                binstr = blocks[cur]['binstr']
                while binstr > 0:
                    if (binstr & 1) == 1: # create a block according to the binary string
                        box = compute_new_bbox(idx, blocks[cur]['bbox_min'], blocks[cur]['bbox_max'])
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': box[0], 'bbox_max': box[1]}
                        blocks.append(block)
                    idx += 1
                    binstr >>= 1
            cur += 1
            if cur >= len(blocks): break

        return [blocks[leaf_idx[i]] for i in range(len(leaf_idx))]


class OctreeOrganizer(OctreeConverter):
    """Prepare the octree array and data of skip blocks given the syntax, so as to enable internal data communications."""

    def __init__(self, octree_cfg, max_num_points, syntax_gt, rw_octree=False, shuffle_blocks=False):

        # Grab the specs for octree partitioning and create an octree converter
        super().__init__(
            octree_cfg['bbox_min'],
            octree_cfg['bbox_max'],
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max'],
        )

        # Set the octree partitioning options
        self.syntax_gt = syntax_gt
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2
        self.shuffle_blocks = shuffle_blocks
        self.infinitesimal = 1e-6

    def get_normalizer(self, bbox_min, bbox_max, pnts=None):
        center = (bbox_min + bbox_max) / 2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):
        if self.rw_octree and os.path.isfile(file_name): # Check whether the point cloud has been converted to octree already
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # Perform octree partitioning
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw, normal)
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # write down the partitioning results
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)

        # Organize the data for batching
        total_cnt = 0
        points_out = np.zeros((self.max_num_points, self.syntax_gt['__len__']), dtype=np.float32)
        normal_out = np.zeros((self.max_num_points, 3), dtype=np.float32) if normal is not None else None
        block_pntcnt = []

        # Shuffle the blocks, only for training
        if self.shuffle_blocks: np.random.shuffle(leaf_idx)

        all_skip = True
        for idx in leaf_idx:
            pnt_start, pnt_end = blocks[idx]['pnt_range'][0], blocks[idx]['pnt_range'][1]
            xyz_slc = slice(pnt_start, pnt_end + 1)
            cnt = pnt_end - pnt_start + 1

            # If we can still add more blocks then continue
            if total_cnt + cnt <= self.max_num_points:
                block_slc = slice(total_cnt, total_cnt + cnt)
                center, scaling = self.get_normalizer(
                    blocks[idx]['bbox_min'], blocks[idx]['bbox_max'], points[xyz_slc, :])
                points_out[block_slc, 0 : points.shape[1]] = points[xyz_slc, :] # x, y, z, and others if exists
                points_out[block_slc, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1] = center # center of the block
                points_out[block_slc, self.syntax_gt['block_scale']] = scaling # scale of the blcok
                points_out[block_slc, self.syntax_gt['block_pntcnt']] = cnt if (blocks[idx]['binstr'] >= 0) else -cnt # number of points in the block
                points_out[total_cnt, self.syntax_gt['block_start']] = 1 if (blocks[idx]['binstr'] >= 0) else -1 # start flag of the block
                if normal is not None: normal_out[block_slc, :] = normal[xyz_slc, :]
                if (blocks[idx]['binstr'] >= 0): all_skip = False
                block_pntcnt.append(cnt)
                total_cnt += cnt
            else: break

        # More stuffs can be returned here, e.g., details about the skip blocks
        return points_out, normal_out, binstrs, np.asarray(block_pntcnt), all_skip
