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
import csv
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

# Load different utilities from pccAI
from pccai.utils.option_handler import BenchmarkOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.bench import *


def aggregate_sequence_log(log_dict_all):
    '''
    Aggregate the dictionaries belonging to the same point cloud seqence as one dictionary, will be used when 
    benchmarking dynamic point cloud sequences 
    '''
    for ckpt in log_dict_all.keys():
        log_dict_ckpt = log_dict_all[ckpt]
        log_dict_ckpt.sort(key=lambda x: x['seq_name'])
        cur_seq_name = ''
        log_dict_ckpt_aggregate=[]

        for idx, log_dict in enumerate(log_dict_ckpt):
            if log_dict['seq_name'].lower() != cur_seq_name: # encounter a new sequence
                cur_seq_name = log_dict['seq_name'].lower()
                log_dict_tmp = { # make a new dictionary, only include keys necessary for MPEG reporting
                    'pc_name': cur_seq_name,
                    'rec_num_points': log_dict['rec_num_points'],
                    'bit_total': log_dict['bit_total'],
                    'd1_psnr': log_dict['d1_psnr'],
                    'seq_cnt': 1
                }
                if 'd2_psnr' in log_dict:
                    log_dict_tmp['d2_psnr'] = log_dict['d2_psnr']
                if 'enc_time' in log_dict:
                    log_dict_tmp['enc_time'] = float(log_dict['enc_time'])
                if 'dec_time' in log_dict:
                    log_dict_tmp['dec_time'] = float(log_dict['dec_time'])
                log_dict_ckpt_aggregate.append(log_dict_tmp)
            else: # update the existing sequence
                log_dict_ckpt_aggregate[-1]['rec_num_points'] += log_dict['rec_num_points']
                log_dict_ckpt_aggregate[-1]['bit_total'] += log_dict['bit_total']
                log_dict_ckpt_aggregate[-1]['d1_psnr'] += log_dict['d1_psnr']
                log_dict_ckpt_aggregate[-1]['seq_cnt'] += 1
                if 'd2_psnr' in log_dict:
                    log_dict_ckpt_aggregate[-1]['d2_psnr'] += log_dict['d2_psnr']
                if 'enc_time' in log_dict:
                    log_dict_ckpt_aggregate[-1]['enc_time'] += float(log_dict['enc_time'])
                if 'dec_time' in log_dict:
                    log_dict_ckpt_aggregate[-1]['dec_time'] += float(log_dict['dec_time'])

        # Take average for each sequence
        for idx, log_dict in enumerate(log_dict_ckpt_aggregate):
            log_dict['d1_psnr'] /= log_dict['seq_cnt']
            if 'd2_psnr' in log_dict:
                log_dict['d2_psnr'] /= log_dict['seq_cnt']
            if 'enc_time' in log_dict:
                log_dict['enc_time'] = str(log_dict['enc_time'])
            if 'dec_time' in log_dict:
                log_dict['dec_time'] = str(log_dict['dec_time'])

        log_dict_all[ckpt] = log_dict_ckpt_aggregate
    return None


def flatten_ckpt_log(log_dict_all):
    '''
    The original log_dict_all is a dictionary indexed by the ckpts, then log_dict_all[ckpt] is a list of several
    dictionaries, each correspoing to the results of a inference test. This function flatten log_dict_all, so
    the output log_dict_all_flat is a list of dicionaries, and sorted by the pc_name (1st key) and bit_total (2nd key)
    '''
    log_dict_all_flat = []
    for ckpt, log_dict_ckpt in log_dict_all.items():
        for log_dict in log_dict_ckpt:
            log_dict['ckpt'] = ckpt
        log_dict_all_flat += log_dict_ckpt
    log_dict_all_flat.sort(key=lambda x: (x['pc_name'], int(x['bit_total']))) # perform sorting with two keys
    return log_dict_all_flat
    

def gen_mpeg_report(log_dict_all, mpeg_report_path, compute_d2, mpeg_report_sequence):
    """Generate the MPEG reporting CSV file"""

    # Parse the MPEG reporting template
    mpeg_seqname_file = os.path.join(os.path.split(__file__)[0], '..', 'assets', 'mpeg_test_seq.txt')
    with open(mpeg_seqname_file) as f:
        lines = f.readlines()
    mpeg_sequence_name = [str[:-1] for str in lines]

    # Preprocessing to log_dict_all
    if mpeg_report_sequence:
        aggregate_sequence_log(log_dict_all)
    log_dict_all = flatten_ckpt_log(log_dict_all)

    # Write down CSV file for MPEG reporting
    mpeg_report_dict_list = []
    for log_dict in log_dict_all:
        pc_name = os.path.splitext(log_dict['pc_name'])[0].lower()
        if pc_name[-2:] == '_n':
            pc_name = pc_name[:-2]
        if pc_name in mpeg_sequence_name: # found an MPEG sequence
            mpeg_report_dict = {
                'sequence': pc_name, # sequence
                'numOutputPointsT': log_dict['rec_num_points'], # numOutputPointsT
                'numBitsGeoEncT': log_dict['bit_total'], # numBitsGeoEncT
                'd1T': log_dict['d1_psnr'] # d1T,
            }
            if compute_d2:
                mpeg_report_dict['d2T'] = log_dict['d2_psnr'] # d2T

            # Encoding/decoding time
            if 'enc_time' in log_dict:
                mpeg_report_dict['encTimeT'] = log_dict['enc_time']
            if 'dec_time' in log_dict:
                mpeg_report_dict['decTimeT'] = log_dict['dec_time']
            mpeg_report_dict_list.append(mpeg_report_dict)

    # Write the CSV file according to the aggregated statistics
    mpeg_report_header = ['sequence', 'numOutputPointsT', 'numBitsGeoEncT', 'd1T', 'd2T', 'encTimeT', 'decTimeT']
    with open(mpeg_report_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=mpeg_report_header)
        writer.writeheader()
        writer.writerows(mpeg_report_dict_list)
    if len(mpeg_report_dict_list) > 0:
        logger.log.info('CSV file for MPEG reporting: %s' % mpeg_report_path)


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
    log_dict_all = benchmark_checkpoints(opt)

    # Create the MPEG reporting CSV file if needed
    if opt.mpeg_report is not None:
        gen_mpeg_report(
            log_dict_all=log_dict_all, 
            mpeg_report_path=os.path.join(opt.exp_folder, opt.mpeg_report), 
            compute_d2=opt.compute_d2,
            mpeg_report_sequence=opt.mpeg_report_sequence
        )
    logger.log.info('Benchmarking session %s finished.\n' % opt.exp_name)
    logger.destroy_logger()

