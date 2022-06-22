# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Handle all the input argumants during training, testing, benchmarking, etc.

import pccai.utils.logger as logger
import argparse
import os

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif val.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expect a Boolean value.')


class BasicOptionHandler():
    """A class that includes the basic options sharing among all phases."""

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_options(parser)
        self.parser = parser

    def add_options(self, parser):

        # What do you want to do
        parser.add_argument('--exp_name', type=str, default='experiment_name', help='Name of the experiment, result folder is created based on this name.')

        # What are your ingredients
        parser.add_argument('--net_config', type=str, default='', help='Network configuration in YAML.')
        parser.add_argument('--optim_config', type=str, default='', help='Optimization configuration in YAML.')
        parser.add_argument('--hetero', type=str2bool, nargs='?', const=True, default=False, help='Whether to use the heterogeneous batching mode.')
        parser.add_argument('--checkpoint', type=str, default='', help='Load an existing checkpoint.')

        # How do you cook
        parser.add_argument('--alpha', type=float, default=None, help='Weight for distortion in R-D optimization, can overwrite the one in the YAML config.')
        parser.add_argument('--beta', type=float, default=None, help='Weight for bit-rate in R-D optimization, can overwrite the one in the YAML config.')
        parser.add_argument('--seed', type=float, default=None, help='Set random seed for reproducibility')

        # Logging options
        parser.add_argument('--result_folder', type=str, default='results', help='Indicate the result folder.')
        parser.add_argument('--log_file', type=str, default='', help='Log file name.')
        parser.add_argument('--log_file_only', type=str2bool, nargs='?', const=True, default=False, help='Only prints to the log file if set True.')
        parser.add_argument('--print_freq', type=int, default=20, help='Frequency of displaying results.')
        parser.add_argument('--pc_write_freq', type=int, default=50, help='Frequency of writing down the point cloud, use tensorboard to write during training, write "ply" file proint cloud during testing.')  
        parser.add_argument('--tf_summary', type=str2bool, nargs='?', const=True, default=False, help='Whether to use tensorboard for log.')
        return parser

    def parse_options(self):
        opt, _ = self.parser.parse_known_args()
        opt.exp_folder = os.path.join(opt.result_folder, opt.exp_name)
        return opt

    def print_options(self, opt):
        message = ''
        message += '\n----------------- Input Arguments ---------------\n'
        # For k, v in sorted(vars(opt).items()):
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.log.info(message)


class TrainOptionHandler(BasicOptionHandler):
    """A class that includes specific options for training."""

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--train_data_config', type=str, nargs='+', required=True, help='Training data configuration in YAML.')
        parser.add_argument('--val_data_config', type=str, nargs='+', default='', help='Validataion data configuration in YAML.')
        parser.add_argument('--checkpoint_optim_config', type=str2bool, nargs='?', const=True, default=False, help='Whether to load the optimizers and schedulers from checkpoint.')
        parser.add_argument('--checkpoint_epoch_state', type=str2bool, nargs='?', const=True, default=False, help='Whether to load the epoch state from the checkpoint.')
        parser.add_argument('--save_checkpoint_freq', type=int, default=2, help='Frequency of saving the trained model.')
        parser.add_argument('--save_checkpoint_max', type=int, default=10, help='Maximum number of check points to be saved.')
        parser.add_argument('--save_checkpoint_prefix', type=str, default='epoch_', help='Prefix of the check points file names. ')
        parser.add_argument('--val_freq', type=int, default=-1, help='Frequency of validation with the validation set, <=0 means no validation.')
        parser.add_argument('--val_print_freq', type=int, default=20, help='Frequency of displaying results during validation.')
        parser.add_argument('--lr', type=float, default=None, help='Learning rate of the main parameters, can overwrite the one in the YAML config.')
        parser.add_argument('--lr_aux', type=float, default=None, help='Learning rate of the aux parameters, can overwrite the one in the YAML config.')
        parser.add_argument('--fix_modules', type=str, nargs='+', default='', help='Names of the fixed modules during training.')
        parser.add_argument('--ddp', type=str2bool, nargs='?', const=True, default=False, help='Whether to DPP mode or not.')
        parser.add_argument('--master_address', type=str, default='localhost', help='Master address of DDP.')
        parser.add_argument('--master_port', type=int, default=29500, help='Master port of DPP.')

        # You can add your method-specific parameters here if necessary. They can be passed to the loaded YAML configs before training.
        # Check how alpha, beta and lr are overwritten in pipelines/train.py as examples.
        return parser


class TestOptionHandler(BasicOptionHandler):
    """A class that includes specific options for tesing."""

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--checkpoint_net_config', type=str2bool, nargs='?', const=True, default=False, help='Whether to load the model configuration from the checkpoint, if yes, net_config will be ignored.')
        parser.add_argument('--test_data_config', type=str, nargs='+', required=True, help='Test data configuration in YAML.')
        parser.add_argument('--gen_bitstream', type=str2bool, nargs='?', const=True, default=False, help='Generate the actual bitstream or not.')
        parser.add_argument('--pc_write_prefix', type=str, default='', help='Prefix when writing down the point clouds.')  
        return parser


class BenchmarkOptionHandler(BasicOptionHandler):
    """A class that includes specific options for benchmarking."""

    def add_options(self, parser):
        parser = BasicOptionHandler.add_options(self, parser)
        parser.add_argument('--checkpoints', type=str, nargs='+', default=None, help='Specify several existing checkpoints.')
        parser.add_argument('--checkpoint_net_config', type=str2bool, nargs='?', const=True, default=True, help='Whether to load the model configuration from the checkpoint, if yes, net_config will be ignored.')
        parser.add_argument('--codec_config', type=str, required=True, help='Codec configuration in YAML.')
        parser.add_argument('--input', type=str, nargs='+', required=True, help='A list of folders containing the point clouds to be tested, or simply just a ply file.')
        parser.add_argument('--peak_value', type=int, nargs='+', required=True, help='Peak value(s) for computing the D1 and D2 metrics. If only one value is provided, it will be used for the whole test; otherwise peak values for every point clouds need to be given.')
        parser.add_argument('--bit_depth', type=int, nargs='+', required=True, help='Bit-depth value(s) of the point cloud(s) to be tested. If only one value is provided, it will be used for the whole test; otherwise bit-depths for every point clouds need to be given.')
        parser.add_argument('--remove_compressed_files', type=str2bool, nargs='?', const=True, default=True, help='Whether to remove the compressed files.')
        parser.add_argument('--compute_d2', type=str2bool, nargs='?', const=True, default=False, help='Whether to compute the D2 metric.')
        parser.add_argument('--mpeg_report', type=str, default=None, help='Write the results for MPEG reporting in the CSV format.')
        parser.add_argument('--mpeg_report_sequence', type=str2bool, nargs='?', const=True, default=False, help='If true, create MPEG report in the CSV format by viewing the inputs as point cloud sequences.')
        parser.add_argument('--write_prefix', type=str, default='', help='Prefix when writing down the point clouds and the bitstreams.')
        parser.add_argument('--slice', type=int, default=None, help='Slicing parameter.')
        return parser