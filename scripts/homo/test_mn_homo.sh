# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


PY_NAME="${HOME_DIR}/experiments/test.py"

# Main configurations
HETERO="False"
OPTIM_CONFIG="${HOME_DIR}/config/optim_config/optim_cd_canonical.yaml"
TEST_DATA_CONFIG="${HOME_DIR}/config/data_config/modelnet_simple.yaml test_cfg"
CHECKPOINT="${HOME_DIR}/results/train_mn_homo/epoch_newest.pth"
CHECKPOINT_NET_CONFIG="True"
GEN_BITSTREAM="True"

# Logging settings
PRINT_FREQ="10"
PC_WRITE_FREQ="20"
TF_SUMMARY="False"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
