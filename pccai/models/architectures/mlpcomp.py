# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import torch
import torch.nn as nn
from pccai.models.modules.get_modules import get_module_class

# Make an attempt to load CompressAI
try:
    from compressai.models.priors import CompressionModel
    found_compressai = True
except ModuleNotFoundError:
    found_compressai = False
    CompressionModel = nn.Module


class MlpCompression(CompressionModel):

    """A simple compression architecture with MLP Decoder."""

    def __init__(self, net_config, syntax):

        if found_compressai: 
            super().__init__(net_config['entropy_bottleneck'], False)
        else:
            super().__init__()
        self.encoder = get_module_class(net_config['cw_gen']['model'], syntax.hetero)(net_config['cw_gen'], syntax=syntax)
        decoder_class_name = net_config['pc_gen'].get('model', 'mlpdecoder') # use "mlpdecoder" by default
        self.decoder = get_module_class(decoder_class_name, syntax.hetero)(net_config['pc_gen'], syntax=syntax)
        self.syntax = syntax
        self.compression = net_config.get('compression', True) and found_compressai
        self.entropy_bottleneck_channels = net_config['entropy_bottleneck']

    def forward(self, x):

        y = self.encoder(x)
        if self.compression:
            y_hat, y_likelihoods = self.entropy_bottleneck(y[:, :self.entropy_bottleneck_channels].unsqueeze(-1).unsqueeze(-1)) # remove the metadata
            y_hat = y_hat.squeeze(-1).squeeze(-1)
            y_likelihoods = y_likelihoods.squeeze(-1).squeeze(-1)
        else:
            y_hat = y[:, :self.entropy_bottleneck_channels]
        x_hat = self.decoder(torch.hstack((y_hat, y[:, self.entropy_bottleneck_channels:]))) # also pass the metadata to the decoder when hetero is on

        output = {"x_hat": x_hat}
        if self.compression:
            output["likelihoods"]={"y": y_likelihoods}

        return output


    def compress(self, x):
        """Performs actual compression with learned statistics of the entropy bottleneck, consumes one point cloud at a time."""

        assert found_compressai
        y = self.encoder(x)
        y_strings = self.entropy_bottleneck.compress(y[:, :self.entropy_bottleneck.channels].unsqueeze(-1).unsqueeze(-1))
        meta_data = y[:, self.entropy_bottleneck.channels:] if self.syntax.hetero else None 

        # "width" and "height" of the codeword are both one
        return {"strings": [y_strings], "shape": torch.Size([1, 1])}, meta_data # meta data also returned


    def decompress(self, strings, shape, meta_data=None):
        """Performs actual decompression with learned statistics of the entropy bottleneck, consumes one point cloud at a time."""

        assert found_compressai and isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape).squeeze(-1).squeeze(-1) # get back the codeword

        if self.syntax.hetero:
            if meta_data is not None:
                y_hat_full = torch.hstack((y_hat, meta_data.squeeze(-1).squeeze(-1)))
                x_hat = self.decoder(y_hat_full)
            else:
                x_hat = self.decoder(y_hat)
            meta_data = x_hat[:, self.syntax.syntax_rec['xyz'][1] + 1:] # this is the new meta_data
            x_hat = x_hat[:, self.syntax.syntax_rec['xyz'][0] : self.syntax.syntax_rec['xyz'][1] + 1]
        else:
            x_hat = self.decoder(y_hat)
            x_hat = x_hat.squeeze(0)
            meta_data = None
        
        return x_hat, meta_data # also return meta data
