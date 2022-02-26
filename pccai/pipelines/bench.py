# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Benchmarking PCC models, this version is dedicated for lossy compression

import time
import os
import numpy as np
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import glob
import pprint
import open3d as o3d

# Load different utilities from pccAI
from pccai.utils.syntax import SyntaxGenerator
from pccai.utils.pc_metric import compute_metrics
from pccai.utils.pc_write import pc_write_o3d
from pccai.codecs.utils import get_codec_class
from pccai.models.pcc_models import get_architecture_class
import pccai.utils.logger as logger


def benchmark_checkpoints(opt):
    """Benchmarking several networks with the same architecture."""

    logger.log.info("%d GPU(s) will be used for benchmarking." % torch.cuda.device_count())
    opt.phase = 'deploy'
    device = torch.device("cuda:0")
    avg_metrics_all = []

    tmp_folder = './tmp'
    os.makedirs(tmp_folder, exist_ok=True)    
    writer = SummaryWriter(comment='_' + opt.exp_name) if opt.tf_summary else None # tensorboard writer to draw R-D curves in real time
    writer_step_resolution = 1e3 if opt.tf_summary else None

    for filename_ckpt in opt.checkpoints:

        checkpoint = torch.load(filename_ckpt)
        if opt.checkpoint_net_config == True:
            opt.net_config = checkpoint['net_config']
            logger.log.info("Model config loaded from check point %s." % opt.checkpoint)
            logger.log.info(opt.net_config)
        syntax = SyntaxGenerator(opt=opt)

        # Construct the PCC model
        architecture_class = get_architecture_class(opt.net_config['architecture'])
        pccnet = architecture_class(opt.net_config['modules'], syntax)

        # Load the network weights
        state_dict = checkpoint['net_state_dict']
        for _ in range(len(state_dict)):
            k, v = state_dict.popitem(False)
            state_dict[k[len('.pcc_model'):]] = v
        pccnet.load_state_dict(state_dict)
        pccnet.to(device)
        pccnet.eval()
        codec = get_codec_class(opt.codec_config['codec'])(opt.codec_config, pccnet, syntax) # initialize a codec
        logger.log.info("Model weights loaded from check point %s.\n" % filename_ckpt)

        # Gather all the point cloud files to be tested
        pc_file_list=[]
        for item in opt.input:
            if item.lower()[-4:] == '.ply':
                pc_file_list.append(item)
            else:
                pc_file_list += list(glob.iglob(item + '/**/*.ply', recursive=True))
        
        # Start the benchmarking
        log_dicts = []
        t = time.monotonic()
        for idx, pc_file in enumerate(pc_file_list):

            # Load the point cloud
            pc_raw = np.asarray(o3d.io.read_point_cloud(pc_file).points)
            pnt_cnt = pc_raw.shape[0]
            pc_name = os.path.splitext(os.path.split(pc_file)[1])[0]

            with torch.no_grad():
                # Encode pc_raw with pccnet, obtain compressed_files
                compressed_files, stat_dict_enc = codec.compress(pc_raw, tag=os.path.join(tmp_folder, pc_name + '_' + opt.exp_name))

                # Decode compressed_files with pccnet, obtain pc_rec
                pc_rec, stat_dict_dec = codec.decompress(compressed_files)

            # Evaluation
            filesize = np.sum([os.stat(f).st_size for f in compressed_files])
            log_dict = {'num_points': pnt_cnt, # compute the metrics then log the results
                        'size': filesize,
                        'bpp': filesize * 8 / pnt_cnt}
            log_dict.update(compute_metrics(pc_file, pc_rec, opt.peak_value, opt.compute_d2))
            log_dicts.append(log_dict)
            if opt.remove_compressed_files:
                for f in compressed_files: os.remove(f)

            # log current metrics if needed
            if opt.print_freq > 0 and idx % opt.print_freq == 0:
                message = '    id: %d/%d, ' % (idx, len(pc_file_list))
                message += 'name: %s, ' % os.path.basename(pc_file)
                message += 'num_points: %d, ' % pnt_cnt
                message += 'rec_num_points: %d, ' % pc_rec.shape[0]
                message += 'bit_total: %d, ' % (filesize * 8)
                message += 'bpp: %f, ' % log_dict['bpp']
                message += 'd1_psnr: %f, ' % log_dict['d1_psnr']
                if opt.compute_d2: message += 'd2_psnr: %f, ' % log_dict['d2_psnr']

                for k, v in stat_dict_enc.items():
                    message += '%s: %s, ' % (k, v)
                for k, v in stat_dict_dec.items():
                    message += '%s: %s, ' % (k, v)
                logger.log.info(message[:-2])
                if opt.print_all_metrics: logger.log.info(pprint.pformat(log_dict))

            # Write down the point cloud if needed
            if opt.pc_write_freq > 0 and idx % opt.pc_write_freq == 0:
                filename_rec = os.path.join(opt.exp_folder, opt.write_prefix + pc_name + "_rec.ply")
                pc_write_o3d(pc_rec, filename_rec)

        elapse = time.monotonic() - t

        # Compute the average metrics for the checkpoint
        all_metrics = [(log_dict['bpp'], log_dict['d1_psnr'], 
            log_dict['d2_psnr'] if opt.compute_d2 else -1) for log_dict in log_dicts]
        avg_bpp, avg_d1_psnr, avg_d2_psnr = np.mean(np.array(all_metrics), axis=0).tolist()
        avg_metrics = {'avg_bpp': avg_bpp, 'avg_d1_psnr': avg_d1_psnr}
        if avg_d2_psnr > 0: avg_metrics['avg_d2_psnr'] = avg_d2_psnr
    
        # Log current metrics for the check point 
        message = 'Compression metrics --- time: %f, ' % elapse
        for k, v in avg_metrics.items(): message += 'avg_%s: %f, ' % (k, v)
        logger.log.info(message[:-2] + '\n')
        if opt.tf_summary:
            writer.add_scalar('d1_psnr-bpp', avg_metrics['avg_d1_psnr'], avg_metrics['avg_bpp'] * writer_step_resolution)
            if 'avg_d2_psnr' in avg_metrics.keys():
                writer.add_scalar('d2_psnr-bpp', avg_metrics['avg_d2_psnr'], avg_metrics['avg_bpp'] * writer_step_resolution)

        avg_metrics_all.append(avg_metrics)

    if opt.tf_summary: writer.close()
    return avg_metrics_all


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