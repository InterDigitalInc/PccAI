# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Test the point cloud compression model, this is to verify the loss on the datasets but not for actual compression

import time
import os
import yaml
import torch
import numpy as np

# Load different utilities from pccAI
from pccai.models import PccModelWithLoss
from pccai.dataloaders.point_cloud_dataset import point_cloud_dataloader
from pccai.utils.syntax import SyntaxGenerator
from pccai.utils.misc import pc_write_o3d, load_state_dict_with_fallback
import pccai.utils.logger as logger


def test_one_epoch(pccnet, dataset, dataloader, syntax, gen_bitstream, print_freq, pc_write_freq, pc_write_prefix, exp_folder=None):
    """Test one epoch with the given model, the specified loss function, and the given dataset, etc."""

    # Perform testing of one epoch
    avg_loss = {}
    avg_real = {'xyz_loss': 0, 'bpp_loss': 0, 'loss': 0} if gen_bitstream else None
    len_data = len(dataloader)
    batch_id = None
    if syntax.hetero:
        syntax_gt = syntax.syntax_gt
        syntax_rec = syntax.syntax_rec

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
            message = '    batch count: %d/%d, ' % (batch_id, len_data)
            for k, v in loss.items(): message += '%s: %f, ' % (k, v)
            logger.log.info(message[:-2])

        # Write down the point cloud if needed
        if pc_write_freq > 0 and batch_id % pc_write_freq == 0:
            filename_gt = os.path.join(exp_folder, pc_write_prefix + str(batch_id) + "_gt.ply")
            filename_rec = os.path.join(exp_folder, pc_write_prefix + str(batch_id) + "_rec.ply")
            if syntax.hetero:
                pc_gt = points[0][points[0][:, syntax_gt['block_pntcnt']] > 0, syntax_gt['xyz'][0] : syntax_gt['xyz'][1] + 1] # take the first point cloud, only keep the blocks with "transform" mode
                pc_rec = output['x_hat'][torch.cumsum(output['x_hat'][:, syntax_rec['pc_start']], dim=0) == 1, 
                    syntax_rec['xyz'][0] : syntax_rec['xyz'][1] + 1]
                pc_write_o3d(pc_gt, filename_gt)
                pc_write_o3d(pc_rec, filename_rec)
            else:
                coloring = np.ones((output['x_hat'][0].shape[0], 3)) * 0.5 # set the colors as 0.5
                normals = output.get('dxyz_dw_n', None)
                if normals is not None:
                    normals = normals[0].contiguous().data.cpu().detach().numpy()
                pc_write_o3d(points[0], filename_gt)
                pc_write_o3d(output['x_hat'][0], filename_rec, coloring, normals=normals)

        # Perform REAL compression, this part is useful under the heterogeneous mode
        if gen_bitstream:
            # Compress then decompress
            with torch.no_grad():
                cmp_out, meta_data = pccnet.pcc_model.compress(points) # compression
                rec_real, meta_data = pccnet.pcc_model.decompress(cmp_out['strings'], cmp_out['shape'], meta_data) # decompression
                if syntax.hetero:
                    rec_real = torch.hstack([rec_real, meta_data])
                elif len(rec_real.shape) == 2:
                        rec_real = rec_real.unsqueeze(0)
        
            # Compute loss and log the results
            bpp_loss_batch = 0 # bit per point for current batch
            for i in range(len(cmp_out['strings'][0])):
                bpp_loss_batch += len(cmp_out['strings'][0][i]) * 8
            if syntax.hetero:
                bpp_loss_batch /= torch.sum(points[:, :, syntax_gt['block_pntcnt']] > 0)
            else:
                bpp_loss_batch /= dataloader.batch_size * dataset.num_points
            xyz_loss_batch = {}
            pccnet.loss.xyz_loss(xyz_loss_batch, points, output) # distortion loss for current batch
            xyz_loss_batch = xyz_loss_batch['xyz_loss'].item()
            real_loss_batch = pccnet.loss.alpha * xyz_loss_batch + pccnet.loss.beta * bpp_loss_batch
            avg_real['bpp_loss'] += bpp_loss_batch
            avg_real['xyz_loss'] += xyz_loss_batch
            avg_real['loss'] += real_loss_batch
            if batch_id % print_freq == 0:
                logger.log.info('        real stat. ---- bpp_loss: %f, xyz_loss: %f, loss: %f' % (bpp_loss_batch, xyz_loss_batch, real_loss_batch))
            
            # Write down the point cloud if needed
            if pc_write_freq > 0 and batch_id % pc_write_freq == 0: # write point clouds if needed
                filename_rec_real = os.path.join(exp_folder, pc_write_prefix + str(batch_id) + "_rec_real.ply")
                if syntax.hetero:
                    pc_rec_real = rec_real[torch.cumsum(rec_real[:, syntax_rec['pc_start']], dim=0) == 1, 
                        syntax_rec['xyz'][0] : syntax_rec['xyz'][1] + 1]
                    pc_write_o3d(pc_rec_real, filename_rec_real)
                else:
                    pc_write_o3d(rec_real[0], filename_rec_real)

    # Log the results
    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # the average loss

    # Log the results if REAL compression has performed
    if gen_bitstream:
        for k in avg_real.keys(): avg_real[k] = avg_real[k] / (batch_id + 1) # the average loss

    return avg_loss, avg_real


def test_pccnet(opt):
    """Test a point cloud compression network. This is not for actual point cloud compression but for the purpose of testing the trained networks."""

    logger.log.info("%d GPU(s) will be used for testing." % torch.cuda.device_count())
    opt.phase = 'test'

    # Load an existing check point
    checkpoint = torch.load(opt.checkpoint)
    if opt.checkpoint_net_config == True:
        opt.net_config = checkpoint['net_config']
        logger.log.info("Model config loaded from check point %s." % opt.checkpoint)
        logger.log.info(opt.net_config)
    syntax = SyntaxGenerator(opt)

    pccnet = PccModelWithLoss(opt.net_config, syntax, opt.optim_config['loss_args'])
    state_dict = checkpoint['net_state_dict']
    for _ in range(len(state_dict)):
        k, v = state_dict.popitem(False)
        state_dict[k[len('.pcc_model'):]] = v
    load_state_dict_with_fallback(pccnet.pcc_model, state_dict)
    logger.log.info("Model weights loaded from check point %s.\n" % opt.checkpoint)
    device = torch.device("cuda:0")
    pccnet.to(device)
    pccnet.eval() # to let the noise add to the codeword, should NOT set it to evaluation mode

    # Miscellaneous configurations
    test_dataset, test_dataloader = point_cloud_dataloader(opt.test_data_config, syntax) # configure the datasets

    # Start the testing process
    t = time.monotonic()
    avg_loss, avg_real = test_one_epoch(pccnet, test_dataset, test_dataloader, syntax,
        opt.gen_bitstream, opt.print_freq, opt.pc_write_freq, opt.pc_write_prefix, opt.exp_folder)
    elapse = time.monotonic() - t

    # Log the testing result
    message = 'Validation --- time: %f, ' % elapse
    for k, v in avg_loss.items(): message += 'avg_%s: %f, ' % (k, v)
    logger.log.info(message[:-2])
    if opt.gen_bitstream:
        message = 'real stat --- '
        for k, v in avg_real.items(): message += 'avg_%s: %f, ' % (k, v)
        logger.log.info(message[:-2])

    return avg_loss


def load_test_config(opt):
    """Load all the configuration files for testing."""

    # Load the test data configuration
    with open(opt.test_data_config[0], 'r') as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)
    opt.test_data_config[0] = test_data_config

    # Load the optimization configuration
    with open(opt.optim_config, 'r') as file:
        optim_config = yaml.load(file, Loader = yaml.FullLoader)
        if opt.alpha is not None:
            optim_config['loss_args']['alpha'] = opt.alpha
        else:
            logger.log.info('alpha from optim config: ' + str(optim_config['loss_args']['alpha']))
        if opt.beta is not None:
            optim_config['loss_args']['beta'] = opt.beta
        else:
            logger.log.info('beta from optim config: ' + str(optim_config['loss_args']['beta']))
    opt.optim_config = optim_config

    # Load the network configuration
    if opt.net_config != '':
        with open(opt.net_config, 'r') as file:
            net_config = yaml.load(file, Loader=yaml.FullLoader)
        opt.net_config = net_config

    return opt


if __name__ == "__main__":

    logger.log.error('Not implemented.')