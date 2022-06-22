# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.

'''
Merge CSV files for MPEG reporting purpose.
Usage: python ./utils/merge_csv.py --input_files file_1.csv file_2.csv --output_file file_3.csv
'''

import argparse
import csv
import os


def main(opt):

    # Read the input CSV files and sort the entries
    log_dict_all = []
    for csv_file in opt.input_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for item in reader:
                log_dict_all.append(dict(item))
    log_dict_all.sort(key=lambda x: (x['sequence'], int(x['numBitsGeoEncT']))) # perform sorting with two keys

    # Write the merged CSV file
    mpeg_report_header = ['sequence', 'numOutputPointsT', 'numBitsGeoEncT', 'd1T', 'd2T', 'encTimeT', 'decTimeT']
    with open(opt.output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=mpeg_report_header)
        writer.writeheader()
        writer.writerows(log_dict_all)


def add_options(parser):

    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='File name of the input image.')
    parser.add_argument('--output_file', type=str, required=True, help='File name of the output image.')

    return parser


if __name__ == "__main__":

    # Initialize parser with basic options
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_options(parser)
    opt, _ = parser.parse_known_args()
    main(opt)