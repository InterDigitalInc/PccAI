# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Defines and generates the internal syntax and status, which serves the heterogeneous mode and marks the module phase

def gen_syntax_gt(hetero):
    if hetero:
        syntax_gt = {
            '__len__': 10,
            'xyz': [0, 2],
            'block_pntcnt': 3,
            'block_center': [4, 6],
            'block_scale': 7,
            'block_start': 9,
        }
    else:
        syntax_gt = None
    return syntax_gt


class SyntaxGenerator():
    """Generate the syntax for internal data and module status communications."""

    def __init__(self, opt):
        self.hetero = opt.hetero
        self.phase = opt.phase
        self.generate_syntax_gt()
        self.generate_syntax_rec()
        self.generate_syntax_cw(opt.net_config)

    def generate_syntax_gt(self, **kwargs):
        """xyz have to be arranged at the beginning, the rest can be swapped
        Data syntax: x, y, z, block_pntcnt, block_center, block_scale, block_start
              index: 0, 1, 2,      3,         4 ~ 6,           7,           8
        """
        self.syntax_gt = gen_syntax_gt(self.hetero)

    def generate_syntax_rec(self, **kwargs):
        """xyz have to be arranged at the beginning, the rest can be swapped
        Rec. syntax: x, y, z, pc_start
              index: 0, 1, 2,     3
        """
        if self.hetero:
            self.syntax_rec = {
                '__len__': 10,
                'xyz': [0, 2],
                'block_start': 3,
                'block_center': [4, 6],
                'block_scale': 7,
                'pc_start': 8,
            }
        else: self.syntax_rec = None

    def generate_syntax_cw(self, net_config, **kwargs):
        """Codewords have to be arranged at the beginning, the rest can be swapped
        Code syntax: codeword, block_pntcnt, block_center, block_scale, pc_start
              index:  0 ~ 511,     512,        513 ~ 515,      516,        517
                                   \--------------------  --------------------/
                                                        \/
                                                    meta_data
        """
        if self.hetero:
            len_cw = net_config['modules']['entropy_bottleneck']
            self.syntax_cw = {
                '__len__': len_cw + 7,
                '__meta_idx__': len_cw,
                'cw': [0, len_cw - 1],
                'block_pntcnt': len_cw,
                'block_center': [len_cw + 1, len_cw + 3],
                'block_scale': len_cw + 4,
                'pc_start': len_cw + 5,
            }
        else: self.syntax_cw = None


def syn_slc(syntax, attr):
    """Create a slice from a syntax and a key"""

    syn = syntax[attr]
    return slice(syn[0], syn[1] + 1)