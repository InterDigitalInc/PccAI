# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


PY_NAME="${HOME_DIR}/experiments/train.py"

# Main configurations
HETERO="True"
NET_CONFIG="${HOME_DIR}/config/net_config/mlp_compression.yaml"
OPTIM_CONFIG="${HOME_DIR}/config/optim_config/optim_cd_canonical.yaml"
TRAIN_DATA_CONFIG="${HOME_DIR}/config/data_config/ford_octree.yaml train_cfg"
VAL_DATA_CONFIG="${HOME_DIR}/config/data_config/ford_octree.yaml val_cfg"

# Logging settings
PRINT_FREQ="20"
PC_WRITE_FREQ="-1"
TF_SUMMARY="True"
SAVE_CHECKPOINT_FREQ="1"
SAVE_CHECKPOINT_MAX="10"
VAL_FREQ="5"
VAL_PRINT_FREQ="20"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
