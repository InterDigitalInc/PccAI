# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Make an attempt to load MinkowskiEngine
try:
    import MinkowskiEngine as ME
    found_ME = True
except ModuleNotFoundError:
    found_ME = False

import torch
import torch.nn as nn
from pccai.models.modules.get_modules import get_module_class


class SparseCnnCompression():
    """
    This example shows how pccAI works with MinkowskiEngine and the sparse_collate() function in
    point cloud_dataset.py to operate on sparse 3D tensors A simple compression architecture using 
    sparse convolutions. This is just an incomplete template for reference.
    """

    def __init__(self, net_config, syntax):
        super().__init__(net_config['entropy_bottleneck'], False)

        # initialize necessary modules with get_module_class()
        return None

    def forward(self, coords):

        # Construct coordnates from sparse tensor
        coords = coords[coords[:, 0] != -1]
        coords[0][0] = 0
        coords[:, 0] = torch.cumsum(coords[:,0], 0)
        device = coords.device

        # An example to build a sparse tensor x with the MinkowskiEngine
        if found_ME:
            x = ME.SparseTensor(
                features=torch.ones(coords.shape[0], 1, device=device, dtype=torch.float32),
                coordinates=coords, 
                device=device)

        # TODO: Perform processing to the sparse tensor x

        return None

    def compress(self, x):
        """Performs actual compression with learned statistics of the entropy bottleneck, consumes one point cloud at a time."""
        return None

    def decompress(self, strings, shape, meta_data=None):
        """Performs actual decompression with learned statistics of the entropy bottleneck, consumes one point cloud at a time."""
        return None
