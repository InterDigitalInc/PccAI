# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Benchmarking one or more models

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import random
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

# Load different utilities from pccAI
from pccai.utils.option_handler import BenchmarkOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.bench import * 


if __name__ == "__main__":

    # Parse the options and perform training
    option_handler = BenchmarkOptionHandler()
    opt = option_handler.parse_options()

    # Create a folder to save the models and the log
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)

    # Initialize a global logger then print out all the options
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)
    option_handler.print_options(opt)
    opt = load_benchmark_config(opt)

    # Go with the actual training
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    avg_metrics_all = benchmark_checkpoints(opt)
    logger.log.info('Benchmarking session %s finished.\n' % opt.exp_name)
    logger.destroy_logger()
