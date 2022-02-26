# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import torch.nn as nn
from pccai.optim.utils import get_loss_class

# Import all the architectures to be used
from pccai.models.architectures.mlpcomp import MlpCompression
from pccai.models.architectures.sparsecnncomp import SparseCnnCompression

# List the all the architectures in the following dictionary 
# For a custom architecture, it is recommended to implement a compress() and a decompress() functions that can be called by the codec.
architectures = {
    'mlp_compression': MlpCompression,
    'sparsecnn_compression': SparseCnnCompression,
}


def get_architecture_class(architecture_name):
    architecture = architectures.get(architecture_name.lower(), None)
    assert architecture is not None, f'architecture "{architecture_name}" not found, valid architectures are: {list(architectures.keys())}'
    return architecture


class PccModelWithLoss(nn.Module):
    """A wrapper class for point cloud compression model and its associated loss function."""

    def __init__(self, net_config, syntax, loss_args = None):
        super(PccModelWithLoss, self).__init__()

        # Get the architecture and initilize it
        architecture_class = get_architecture_class(net_config['architecture'])
        self.pcc_model = architecture_class(net_config['modules'], syntax)

        # Get the loss class and initlize it
        if loss_args is not None:
            loss_class = get_loss_class(loss_args['loss'])
            self.loss = loss_class(loss_args, syntax)
    
    def forward(self, data):
        out = self.pcc_model(data)
        if self.loss is not None: out['loss'] = self.loss.loss(data, out)

        return out