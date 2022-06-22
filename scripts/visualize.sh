#!/usr/bin/env bash

# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Rendering settings
FILE="./datasets/ford/Ford_01_q_1mm/Ford_01_vox1mm-0100.ply"
RADIUS=-1
RADIUS_ORIGIN=-1
VIEW_FILE=.

# Begin rendering
python ./utils/visualize.py \
--file_name $FILE \
--output_file . \
--view_file $VIEW_FILE \
--radius $RADIUS \
--radius_origin $RADIUS_ORIGIN \
--window_name $FILE
