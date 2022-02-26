#!/usr/bin/env bash

# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


if [ $# -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=$2
    echo export CUDA_VISIBLE_DEVICES=$2
fi

source './scripts/tmp/'$1
echo $1
echo python ${RUN_ARGUMENTS}
python ${RUN_ARGUMENTS}
