# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Training functions.
# The aux optimizer is for the compatibility of CompressAI (if it is used)

import shutil
import time
import os
import numpy as np
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Load different utilities from PccAI
from pccai.models import PccModelWithLoss
from pccai.optim.utils import configure_optimization
from pccai.utils.syntax import SyntaxGenerator
from pccai.dataloaders.point_cloud_dataset import point_cloud_dataloader
from pccai.utils.misc import load_state_dict_with_fallback
import pccai.utils.logger as logger


def save_checkpoint(pccnet, optimizer, scheduler, epoch_state, aux_optimizer, aux_scheduler, opt, checkpoint_name):
    if opt.ddp and dist.get_rank() != 0: return # in DDP mode, only save checkpoint when rank is 0
    data = {
        'net_state_dict': pccnet.module.state_dict(),
        'net_config': opt.net_config,
        'epoch_state': epoch_state,
    }
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()
    if aux_optimizer is not None:
        data['aux_optimizer_state_dict'] = aux_optimizer.state_dict()
    if aux_scheduler is not None:
        data['aux_scheduler_state_dict'] = aux_scheduler.state_dict()
    if (aux_optimizer is not None) or (aux_scheduler is not None):
        pccnet.module.pcc_model.update()
    torch.save(data, checkpoint_name)


def load_checkpoint(checkpoint_path, with_optim, with_epoch_state, pccnet, epoch_state, optimizer=None,
                    scheduler=None, aux_optimizer=None, aux_scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    
    if with_epoch_state:
        epoch_state.update(checkpoint.get('epoch_state', {}))

    load_state_dict_with_fallback(pccnet.module, checkpoint['net_state_dict'])
    
    logger.log.info("Existing model %s loaded.\n" % (checkpoint_path))
    
    if with_optim: # load the optimizers and schedulers if needed
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if aux_optimizer is not None:
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
        if aux_scheduler is not None:
            aux_scheduler.load_state_dict(checkpoint['aux_scheduler_state_dict'])
        logger.log.info("Optimization parameters loaded.\n")


def train_one_epoch(pccnet, dataloader, optimizer, aux_optimizer, writer, batch_total, opt):
    """Train one epoch with the model, the specified loss, optimizers, scheduler, etc."""

    pccnet.train() # set model to training mode
    avg_loss = {}
    len_data = len(dataloader)
    batch_id = None

    # Iterates the training process
    for batch_id, points in enumerate(dataloader):
        if (points.shape[0] < dataloader.batch_size):
            batch_id -= 1
            break

        points = points.cuda()
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        # Forward and backward for the main loss
        output = pccnet(points)

        loss = output['loss']
        for k, v in loss.items(): loss[k] = torch.mean(v)

        loss['loss'].backward() # Compute gradient
        if opt.optim_config['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(pccnet.parameters(), opt.optim_config['clip_max_norm'])
        optimizer.step() # Update parameters

        # Forward and backward for the auxiliary loss
        if aux_optimizer is not None:
            aux_loss = pccnet.module.pcc_model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        # Log the results
        for k in loss:
            avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()

        if batch_id % opt.print_freq == 0:
            message = '    batch count: %d/%d, iter: %d, ' % (batch_id, len_data, batch_id)
            for k, v in loss.items(): 
                message += '%s: %f, ' % (k, v)
                if writer is not None: writer.add_scalar('batch/' + k, v, batch_total)
            if aux_optimizer is not None:
                message += 'aux_loss: %f, ' % (aux_loss.item())
            logger.log.info(message[:-2])

        # Write down the point cloud, only support homogeneous batching
        if opt.pc_write_freq > 0 and batch_total % opt.pc_write_freq == 0 and opt.hetero == False:
            labels = np.concatenate((np.ones(output['x_hat'].shape[1]), np.zeros(points.shape[1])), axis=0).tolist()
            if writer is not None: writer.add_embedding(torch.cat((output['x_hat'][0, 0:, 0:3], points[0, :, 0:3]), 0), 
                global_step=batch_total, metadata=labels, tag="point_cloud")

        # torch.cuda.empty_cache() # empty cache
        batch_total += 1

    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # the average loss
    return avg_loss, batch_total


def validate_one_epoch(pccnet, dataloader, print_freq):
    """Validate one epoch with the model, the specified loss, etc."""

    pccnet.eval() # set model to evaluation mode
    avg_loss = {}
    len_data = len(dataloader)
    batch_id = None

    # Iterates the validation process
    for batch_id, points in enumerate(dataloader):
        
        if(points.shape[0] < dataloader.batch_size):
            batch_id -= 1
            break
        points = points.cuda()

        # Inference and compute loss
        with torch.no_grad():
            output = pccnet(points)
            loss = output['loss']
            for k, v in loss.items(): loss[k] = torch.mean(v)

        # Log the results
        if batch_id == 0:
            for k, v in loss.items(): avg_loss[k] = v.item()
        else:
            for k, v in loss.items(): avg_loss[k] += v.item()
        if batch_id % print_freq == 0:
            message = '    batch count: %d/%d, iter: %d, ' % (batch_id, len_data, batch_id)
            for k, v in loss.items(): message += '%s: %f, ' % (k, v)
            logger.log.info(message[:-2])

    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # the average loss
    return avg_loss


def train_pccnet(opt):
    """Train a point cloud compression network."""
    
    opt.phase='train'
    syntax = SyntaxGenerator(opt)
    pccnet = PccModelWithLoss(opt.net_config, syntax, opt.optim_config['loss_args'])
    logger.log.info("%d GPU(s) will be used for training." % opt.device_count)
    if opt.ddp:
        # Wrap the autoencoder with DDP
        pccnet.to(opt.device)
        pccnet = DDP(pccnet, device_ids=[opt.device], output_device=[opt.device], find_unused_parameters=True)
    else:
        # Wrap the autoencoder with DP
        pccnet = torch.nn.DataParallel(pccnet)
        pccnet.to(torch.device("cuda:" + str(opt.device))) # 0 is the master

    # Take care of the dataset
    _, train_dataloader = point_cloud_dataloader(opt.train_data_config, syntax, opt.ddp)
    if opt.val_freq > 0:
        _, val_dataloader = point_cloud_dataloader(opt.val_data_config, syntax, opt.ddp)

    # Configure the optimization stuffs
    optimizer, scheduler, aux_optimizer, aux_scheduler = configure_optimization(pccnet, opt.optim_config)
    epoch_state = { 'last_epoch': -1, 'total_epoch': -1 }
    # Load a saved model if given
    if opt.checkpoint != '':
        load_checkpoint(checkpoint_path=opt.checkpoint, with_optim=opt.checkpoint_optim_config, with_epoch_state=opt.checkpoint_epoch_state,
                        pccnet=pccnet, epoch_state=epoch_state, 
                        optimizer=optimizer, scheduler=scheduler,
                        aux_optimizer=aux_optimizer, aux_scheduler=aux_scheduler)
    
        # Fix the weights of the specified modules
        modules = {module_name: param for module_name, param in pccnet.module.named_modules()}
        params = {param_name: param for param_name, param in pccnet.module.named_parameters()}
        for fix_module in opt.fix_modules:
            logger.log.info('Fix the weights of %s' % fix_module)
            param = modules.get(fix_module, params.get(fix_module, None))
            param.requires_grad_(False)

    # Create a tensorboard writer
    writer = SummaryWriter(comment='_' + opt.exp_name) \
        if opt.tf_summary and not(opt.ddp and dist.get_rank() != 0) else None

    # Start the training process
    batch_total = 0
    checkpoint_queue = deque()
    mdl_cnt = sum(p.numel() for p in pccnet.parameters() if p.requires_grad)
    logger.log.info('Model parameter count %d.' % mdl_cnt)

    t = time.monotonic()
    epoch = epoch_state['last_epoch']
    total_epoch = epoch_state['total_epoch']
    while epoch < opt.optim_config['n_epoch']:
        epoch_state['last_epoch'] += 1
        epoch_state['total_epoch'] += 1
        epoch = epoch_state['last_epoch']
        total_epoch = epoch_state['total_epoch']
        if opt.ddp: train_dataloader.sampler.set_epoch(epoch) # This is to entertain DDP

        # Perform training of one epoch
        lr = optimizer.param_groups[0]['lr']
        aux_lr = aux_optimizer.param_groups[0]['lr'] if aux_optimizer is not None else None
        logger.log.info(f'Training at epoch {epoch} (total {total_epoch}) with lr {lr}' +
                        (f' and aux_lr {aux_lr}' if aux_scheduler is not None else ''))
        avg_loss, batch_total = train_one_epoch(pccnet, train_dataloader, optimizer, aux_optimizer, writer, batch_total, opt)

        if scheduler is not None:
            scheduler.step()
        if aux_scheduler is not None:
            aux_scheduler.step()
        elapse = time.monotonic() - t

        # Log the training result
        message = 'Epoch: %d/%d --- time: %f, lr: %f, ' % (epoch, total_epoch, elapse, lr)
        for k, v in avg_loss.items():
            message += 'avg_%s: %f, ' % (k, v)
            if writer is not None: writer.add_scalar('epoch/avg_' + k, v, epoch)
        logger.log.info(message[:-2] + '\n')
        if writer is not None: writer.add_scalar('epoch/learning_rate', lr, epoch)

        # Save the new checkpoint if needed
        if epoch % opt.save_checkpoint_freq == 0 or epoch == opt.optim_config['n_epoch'] - 1:
            checkpoint_name = os.path.join(opt.exp_folder, opt.save_checkpoint_prefix + str(epoch) + '.pth')
            save_checkpoint(pccnet, optimizer, scheduler, epoch_state, aux_optimizer, aux_scheduler, opt, checkpoint_name)

            if not(opt.ddp and dist.get_rank() != 0):
                shutil.copyfile(checkpoint_name, os.path.join(opt.exp_folder, opt.save_checkpoint_prefix + 'newest.pth'))
            logger.log.info('Current checkpoint saved to %s.\n' % (checkpoint_name))

            # Maintain the total checkpoint count
            checkpoint_queue.append((epoch, checkpoint_name))
            if len(checkpoint_queue) > opt.save_checkpoint_max:
                _, pop_checkpoint_name = checkpoint_queue.popleft()
                if os.path.exists(pop_checkpoint_name): os.remove(pop_checkpoint_name)

        # Perform validation of one epoch and log the result
        if opt.val_freq > 0 and epoch % opt.val_freq == 0:
            logger.log.info('Validation at epoch %d' % epoch)
            avg_loss_val = validate_one_epoch(pccnet, val_dataloader, opt.val_print_freq)

            # Log the validation result
            message = 'Validation --- '
            for k, v in avg_loss_val.items():
                message += 'avg_val_%s: %f, ' % (k, v)
                if writer is not None: writer.add_scalar('epoch/avg_val_' + k, v, epoch)
            logger.log.info(message[:-2] + '\n')

    if writer is not None: writer.close()
    return avg_loss


def load_train_config(opt):
    """Load all the configuration files for training."""

    # Load the training and validation data configuration
    with open(opt.train_data_config[0], 'r') as file:
        train_data_config = yaml.load(file, Loader=yaml.FullLoader)
    opt.train_data_config[0] = train_data_config
    if opt.val_data_config != '':
        with open(opt.val_data_config[0], 'r') as file:
            val_data_config = yaml.load(file, Loader=yaml.FullLoader)
        opt.val_data_config[0] = val_data_config

    # Load the optimization configuration
    with open(opt.optim_config, 'r') as file:
        optim_config = yaml.load(file, Loader = yaml.FullLoader)

    # R-D weights and the learning rates are special paramters, can be overwritten from the input parameters
    if opt.alpha is not None:
        optim_config['loss_args']['alpha'] = opt.alpha
    else:
        logger.log.info('alpha from optim config: ' + str(optim_config['loss_args']['alpha']))
    if opt.beta is not None:
        optim_config['loss_args']['beta'] = opt.beta
    else:
        logger.log.info('beta from optim config: ' + str(optim_config['loss_args']['beta']))
    if opt.lr is not None:
        optim_config['main_args']['lr'] = opt.lr
    else:
        logger.log.info('lr from optim config: ' + str(optim_config['main_args']['lr']))

    if 'aux_args' not in optim_config.keys(): # 'main_args' is used if 'aux_args' not specified
        optim_config['aux_args'] = optim_config['main_args']
    if opt.lr_aux is not None: optim_config['aux_args']['lr'] = opt.lr_aux
    opt.optim_config = optim_config

    # Load the network configuration
    with open(opt.net_config, 'r') as file:
        net_config = yaml.load(file, Loader=yaml.FullLoader)

    opt.net_config = net_config

    return opt


if __name__ == "__main__":
    logger.log.error('Not implemented.')
