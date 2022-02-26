# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Import all the modules to be used here
from pccai.models.modules.pointnet import PointNet, PointNetHetero
from pccai.models.modules.mlpdecoder import MlpDecoder, MlpDecoderHetero


def get_module_class(module_name, hetero=False):
    """Retrieve the module classes from the module name."""

    # List all the modules and their string name in this dictionary
    module_dict = {
        'pointnet': [PointNet, PointNetHetero], # pointnet
        'mlpdecoder': [MlpDecoder, MlpDecoderHetero], # mlpdecoder
    }

    module = module_dict.get(module_name.lower(), None)
    assert module is not None, f'module {module_name} was not found, valid modules are: {list(module_dict.keys())}'
    try:
        module = module[hetero]
    except IndexError as e:
        raise Exception(f'module {module_name} is not implemented for hetero={hetero}')

    return module