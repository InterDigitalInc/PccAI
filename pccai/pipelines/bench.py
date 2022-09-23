# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Benchmarking PCC models

import time
import os
import numpy as np
import yaml
import torch
import glob

# Load different utilities from PccAI
from pccai.utils.syntax import SyntaxGenerator
from pccai.utils.pc_metric import compute_metrics
from pccai.utils.misc import pc_read, pc_write, load_state_dict_with_fallback
from pccai.codecs.utils import get_codec_class
from pccai.models.pcc_models import get_architecture_class
import pccai.utils.logger as logger


def create_pccnet(net_config, checkpoint, syntax, device):
    """Build the network model."""

    # Construct the PCC model
    architecture_class = get_architecture_class(net_config['architecture'])
    pccnet = architecture_class(net_config['modules'], syntax)

    # Load the network weights
    state_dict = checkpoint['net_state_dict'].copy()
    for _ in range(len(state_dict)):
        k, v = state_dict.popitem(False)
        state_dict[k[len('.pcc_model'):]] = v
    load_state_dict_with_fallback(pccnet, state_dict)
    pccnet.to(device)
    pccnet.eval()
    logger.log.info("Model weights loaded.")
    return pccnet


def benchmark_checkpoints(opt):
    """Benchmarking several networks with the same architecture."""

    logger.log.info("%d GPU(s) will be used for benchmarking." % torch.cuda.device_count())
    opt.phase = 'deploy'
    device = torch.device("cuda:0")
    log_dict_all = {}
    tmp_folder = './tmp'
    os.makedirs(tmp_folder, exist_ok=True)

    # Gather all the point cloud files to be tested
    pc_file_list=[]
    for item in opt.input:
        if item.lower()[-4:] == '.ply':
            pc_file_list.append(item)
        else:
            pc_file_list += list(glob.iglob(item + '/**/*.ply', recursive=True))
            pc_file_list.sort()

    for filename_ckpt in opt.checkpoints:

        log_dict_ckpt = []
        logger.log.info("Working on checkpoint %s." % filename_ckpt)
        checkpoint = torch.load(filename_ckpt)
        if opt.checkpoint_net_config == True:
            opt.net_config = checkpoint['net_config']
            logger.log.info("Model config loaded from check point.")
            logger.log.info(opt.net_config)
        syntax = SyntaxGenerator(opt=opt)
        pccnet = create_pccnet(opt.net_config, checkpoint, syntax, device)
        
        # Start the benchmarking
        t = time.monotonic()
        for idx, pc_file in enumerate(pc_file_list):

            bit_depth = opt.bit_depth[0 if len(opt.bit_depth) == 1 else idx] # support testing several point clouds with different bit-depths, individual bit_depths need to be provided in this case
            codec = get_codec_class(opt.codec_config['codec'])(opt.codec_config, pccnet, bit_depth, syntax) # initialize the codec

            # Load the point cloud and initialize the log_dict
            pc_raw = pc_read(pc_file)
            log_dict = {
                'pc_name': os.path.split(pc_file)[1],
                'num_points': pc_raw.shape[0],
            }
            if opt.mpeg_report_sequence:
                log_dict['seq_name'] = os.path.basename(os.path.dirname(pc_file))

            with torch.no_grad():
                # Encode pc_raw with pccnet, obtain compressed_files
                compressed_files, stat_dict_enc = codec.compress(pc_raw, tag=os.path.join(tmp_folder, os.path.splitext(log_dict['pc_name'])[0] + '_' + opt.exp_name))

                # Decode compressed_files with pccnet, obtain pc_rec
                if opt.skip_decode == False:
                    pc_rec, stat_dict_dec = codec.decompress(compressed_files)

            # Update the log_dict and compute D1, D2
            log_dict['bit_total'] = np.sum([os.stat(f).st_size for f in compressed_files]) * 8
            log_dict['bpp'] = log_dict['bit_total'] / log_dict['num_points']

            peak_value = opt.peak_value[0 if len(opt.peak_value) == 1 else idx] # support point clouds with different bit-depths, individual peak values need to be provided in this case
            if opt.skip_decode:
                log_dict['d1_psnr'] = -1
                log_dict['d2_psnr'] = -1
                log_dict['rec_num_points'] = -1
            else:
                log_dict.update(compute_metrics(pc_file, pc_rec, peak_value, opt.compute_d2))
                log_dict['rec_num_points'] = pc_rec.shape[0]
            log_dict.update(stat_dict_enc)
            if opt.skip_decode == False:
                log_dict.update(stat_dict_dec)
            log_dict_ckpt.append(log_dict)
            if opt.remove_compressed_files:
                for f in compressed_files: os.remove(f)

            # Log current metrics if needed
            if opt.print_freq > 0 and idx % opt.print_freq == 0:
                message = '    id: %d/%d, ' % (idx + 1, len(pc_file_list))
                for k, v in log_dict.items():
                    message += '%s: %s, ' % (k, str(v))
                logger.log.info(message[:-2])

            # Write down the point cloud if needed
            if opt.pc_write_freq > 0 and idx % opt.pc_write_freq == 0 and opt.skip_decode == False:
                filename_rec = os.path.join(opt.exp_folder, opt.write_prefix + os.path.splitext(log_dict['pc_name'])[0] + "_rec.ply")
                pc_write(pc_rec, filename_rec)

        elapse = time.monotonic() - t
        log_dict_all[filename_ckpt] = log_dict_ckpt

        # Compute the average metrics for this current checkpoint
        basic_metrics = [(log_dict['bpp'], log_dict['bit_total'], log_dict['num_points'], log_dict['d1_psnr'],
            log_dict['d2_psnr'] if opt.compute_d2 else -1) for log_dict in log_dict_ckpt]
        avg_bpp, avg_size, avg_num_points, avg_d1_psnr, avg_d2_psnr = np.mean(np.array(basic_metrics), axis=0).tolist()
        avg_metrics = {'bpp': avg_bpp, 'seq_bpp': avg_size / avg_num_points, 'd1_psnr': avg_d1_psnr}
        if avg_d2_psnr > 0: avg_metrics['d2_psnr'] = avg_d2_psnr

        # Log current metrics for the check point 
        message = 'Compression metrics --- time: %f, ' % elapse
        for k, v in avg_metrics.items(): message += 'avg_%s: %f, ' % (k, v)
        logger.log.info(message[:-2] + '\n')

    return log_dict_all


def load_benchmark_config(opt):
    """Load all the configuration files for benchmarking."""

    # Load the codec configuration
    with open(opt.codec_config, 'r') as file:
        codec_config = yaml.load(file, Loader=yaml.FullLoader)
    if opt.slice is not None:
        codec_config['slice'] = opt.slice
    opt.codec_config = codec_config

    # Load the network configuration
    if opt.net_config != '':
        with open(opt.net_config, 'r') as file:
            net_config = yaml.load(file, Loader=yaml.FullLoader)
        opt.net_config = net_config

    return opt


if __name__ == "__main__":

    logger.log.error('Not implemented.')