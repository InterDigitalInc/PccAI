# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A simple tool to generate temporary scripts which holds the options 

import sys
import os
import random

cur_dir = os.path.dirname(os.path.abspath(__file__))
home_dir=os.path.abspath(os.path.join(cur_dir, '..'))

def main():

    # Create a folder if not exist
    tmp_script_folder = 'tmp'
    tmp_script_path = os.path.join(home_dir, 'scripts', tmp_script_folder)
    if not os.path.exists(tmp_script_path):
        os.makedirs(tmp_script_path)

    # Create the new argument script
    tmp_file_name = 'tmp_'+str(hex(int(random.random() * 1e15)))+'.sh'
    tmp_file = open(os.path.join(home_dir, 'scripts', 'tmp', tmp_file_name), 'w')
    tmp_file.write('HOME_DIR="' + home_dir + '"\n')
    exp_name = os.path.basename(sys.argv[1]).split('.')[0]
    tmp_file.write('EXP_NAME="' + exp_name + '"\n')

    # add the arguments one-by-one
    addline = 'RUN_ARGUMENTS="${PY_NAME} --exp_name ${EXP_NAME} '
    len_addline = len(addline)
    with open(sys.argv[1]) as f:
        args = f.readlines()
        for line in args:
            line = line.lstrip()
            if len(line) > 0 and line[0].isalpha():
                idx = line.find('=')
                opt_name = line[0:idx].upper()
                if opt_name != "PY_NAME" and opt_name != "EXP_NAME":
                    addline += "--" + opt_name.lower() + " ${" + opt_name + "} "
                if opt_name != 'RUN_ARGUMENTS' and opt_name != "EXP_NAME":
                    tmp_file.write(line)
        addline = "\n" + addline[:-1] + '"'
    if len(addline) > len_addline:
        tmp_file.write(addline)

    return tmp_file_name
    

if __name__ == "__main__":

    tmp_file_name = main()
    print(tmp_file_name)
    sys.exit(0)
