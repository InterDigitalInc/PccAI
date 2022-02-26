# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Train a point cloud compression model

import random
import os
import torch
import sys
import socket
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

# multi-processing utilities
import torch.multiprocessing as mp
import torch.distributed as dist

# Load different utilities from pccAI
from pccai.utils.option_handler import TrainOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.train import *


def setup(rank, world_size, master_address, master_port):
    """Setup the DDP processes if necessary, each process will be allocated to one GPU."""

    # Look for an available port first
    tmp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        loc = (master_address, master_port)
        res = tmp_socket.connect_ex(loc)
        if res != 0: break # found a port
        else: master_port += 1

    # initialize the process group
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['MASTER_ADDR'] = master_address
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Destropy all processes."""

    dist.destroy_process_group()


def train_main(device, opt):
    """Main training wrapper."""

    # Initialize a global logger then print out all the options
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)
    option_handler = TrainOptionHandler()
    option_handler.print_options(opt)
    opt = load_train_config(opt)
    opt.device = device
    opt.device_count = torch.cuda.device_count()
    if opt.ddp: setup(device, opt.device_count, opt.master_address, opt.master_port)

    # Go with the actual training
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    avg_loss = train_pccnet(opt)
    logger.log.info('Training session %s finished.\n' % opt.exp_name)
    logger.destroy_logger()
    if opt.ddp: cleanup()


if __name__ == "__main__":

    # Parse the options and perform training
    option_handler = TrainOptionHandler()
    opt = option_handler.parse_options()

    # Create a folder to save the models and the log
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)
    if opt.ddp:
        mp.spawn(train_main, args=(opt,), nprocs=torch.cuda.device_count(), join=True)
    else:
        train_main(0, opt)