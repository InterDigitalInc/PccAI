# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import numpy as np

class PccCodecBase:
    """A base class of PCC codec. User needs to implement the compress() and decompress() method."""

    def __init__(self, codec_config, pccnet, syntax):
        self.translate = codec_config['translate']
        self.scale = codec_config['scale']
        self.hetero = syntax.hetero
        self.phase = syntax.phase
        self.pccnet = pccnet


    def compress(self, points, tag):
        """Compression method."""

        raise NotImplementedError()
    
    
    def decompress(self, file_name):
        """Decompression method."""

        raise NotImplementedError()