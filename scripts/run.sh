#!/usr/bin/env bash

# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


SPEC=$1
LAUNCHER=$2
USE_GPU=$3

TMP_ARGS=`python ./utils/gen_args.py ${SPEC}`
if [[ ${LAUNCHER} == "d" ]]; then
    echo "Launch the job directly."
    ./scripts/config_args.sh ${TMP_ARGS} ${USE_GPU} 2>&1 &
elif [[ ${LAUNCHER} == "f" ]]; then
    echo "Launch the job directly in foreground."
    ./scripts/config_args.sh ${TMP_ARGS} ${USE_GPU} 2>&1
elif [[ ${LAUNCHER} == "s" ]]; then
    echo "Launch the job with slurm."
    source './scripts/tmp/'${TMP_ARGS}
    # Please modify according your needs
    sbatch --job-name=${EXP_NAME} -n 1 -D ${HOME_DIR} --gres=gpu:1 ./scripts/config_args.sh ${TMP_ARGS} 0
else
    echo "No launcher is specified."
fi
