import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='experiment_logs', help='Input path to folder of log files from 12 baseline models')
parser.add_argument('--output_path', type=str, default='experiment_plots', help='Output path to folder to save plots')

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

if not os.path.exists(input_path):
    print(f'Input path {input_path} does not exist, please make sure you are pointing to a folder containing experiment logs')
    sys.exit()

print(f'Reading in experiment logs from {input_path} and saving plots to {output_path}...')

if not os.path.exists(output_path):
    os.makedirs(output_path)

log_files = os.listdir(input_path)
for log_file in log_files:
    print(log_file)
