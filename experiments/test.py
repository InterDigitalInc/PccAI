# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Test a trained point cloud compression model

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import random
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

# Load different utilities from pccAI
from pccai.utils.option_handler import TestOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.test import *


if __name__ == "__main__":

    # Parse the options and perform training
    option_handler = TestOptionHandler()
    opt = option_handler.parse_options()

    # Create a folder to save the models and the log
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)

    # Initialize a global logger then print out all the options
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)
    option_handler.print_options(opt)
    opt = load_test_config(opt)

    # Go with the actual training
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    avg_loss = test_pccnet(opt)
    logger.log.info('Testing session %s finished.\n' % opt.exp_name)
    logger.destroy_logger()
