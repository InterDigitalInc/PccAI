# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import subprocess
import os
import random

from pccai.utils.misc import pc_write
base_path = os.path.split(__file__)[0]

def compute_metrics(gt_file, pc_rec, res, normal=False):
    """Compute D1 and/or D2 with pc_error tool from MPEG"""

    tmp_file_name = os.path.join('./tmp/', 'metric_'+str(hex(int(random.random() * 1e15)))+'.ply')
    rec_file = os.path.join(base_path, '../..', tmp_file_name)
    pc_error_path = os.path.join(base_path, '../..', 'third_party/pc_error')
    pc_write(pc_rec, rec_file)
    cmd = pc_error_path + ' -a '+ gt_file + ' -b '+ rec_file + ' --hausdorff=1 '+ ' --resolution=' + str(res)
    if normal: cmd = cmd + ' -n ' + gt_file
    bg_proc=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    line_b = bg_proc.stdout.readline()

    d1_key = 'mseF,PSNR (p2point):'
    d2_key = 'mseF,PSNR (p2plane):'
    d1_psnr, d2_psnr = None, None
    while line_b:
        line = line_b.decode(encoding='utf-8')
        line_b = bg_proc.stdout.readline()
        idx = line.find(d1_key)
        if idx > 0: d1_psnr = float(line[idx + len(d1_key):])
        if normal:
            idx = line.find(d2_key)
            if idx > 0: d2_psnr = float(line[idx + len(d2_key):])
    os.remove(rec_file)
    return {"d1_psnr": d1_psnr, "d2_psnr": d2_psnr}
