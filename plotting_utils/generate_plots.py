import argparse
import os
import sys
import json

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

train_roc_auc = 'train/hateful_memes/roc_auc'
validation_roc_auc = 'val/hateful_memes/roc_auc'
test_roc_auc = 'test/hateful_memes/roc_auc'
logistics_line_start = 'mmf.trainers.callbacks.logistics : {'

log_files = os.listdir(input_path)
metrics = {}
for log_file in log_files:
    logistics_lines = []
    with open(os.path.join(input_path, log_file)) as f:
        log_lines = f.readlines()
        model_description_line = log_lines[1]
        config_file = model_description_line[model_description_line.find('config=') + 7:model_description_line.find("', 'model")]
        metrics[config_file] = {'train': [], 'validation': [], 'test': []}

        for line in log_lines[2:]:
            if 'mmf.trainers.callbacks.logistics' in line:
                logistics_lines.append(line)

    for logistics_line in logistics_lines:
        if 'Finished run' in logistics_line:
            continue
        current_metrics_string = logistics_line[logistics_line.find(logistics_line_start) + len(logistics_line_start) - 1:]
        current_metrics = json.loads(current_metrics_string)

        progress = current_metrics['progress']
        epoch_num = progress[:progress.find('/')]
        if train_roc_auc in current_metrics:
            metrics[config_file]['train'].append(current_metrics[train_roc_auc])
        elif validation_roc_auc in current_metrics:
            metrics[config_file]['validation'].append(current_metrics[validation_roc_auc])
        elif test_roc_auc in current_metrics:
            metrics[config_file]['test'].append(current_metrics[test_roc_auc])

print(metrics)
