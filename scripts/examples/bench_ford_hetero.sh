# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
HETERO="True"
CHECKPOINTS="${HOME_DIR}/results/train_ford_hetero/epoch_newest.pth"
CHECKPOINT_NET_CONFIG="True"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/ford.yaml"
INPUT="${HOME_DIR}/datasets/ford/Ford_02_q_1mm ${HOME_DIR}/datasets/ford/Ford_03_q_1mm"
COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv" # generate the CSV file for MPEG reporting
MPEG_REPORT_SEQUENCE="True" # view the input point clouds as sequences
WRITE_PREFIX="compress_"
PEAK_VALUE="30000"
BIT_DEPTH="18"

# Logging settings
PRINT_FREQ="1"
PC_WRITE_FREQ="-1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
