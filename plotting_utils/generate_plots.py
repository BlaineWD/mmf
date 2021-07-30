import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='experiment_logs', help='Input path to folder of log files from 12 baseline models')
parser.add_argument('--output_path', type=str, default='experiment_plots', help='Output path to folder to save plots')

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

if not os.path.exists(input_path):
    print(f'Input path {input_path} does not exist, please make sure you are pointing to a folder containing experiment logs')
    sys.exit()


if not os.path.exists(output_path):
    os.makedirs(output_path)

roc_auc_key = 'hateful_memes/roc_auc'
train_roc_auc = f'train/{roc_auc_key}'
validation_roc_auc = f'val/{roc_auc_key}'
test_roc_auc = f'test/{roc_auc_key}'
logistics_line_start = 'mmf.trainers.callbacks.logistics : {'

log_files = os.listdir(input_path)
metrics = {}
epoch_step = None
print(f'Reading in experiment logs from {input_path}...')
for log_file in tqdm(log_files):
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

        if epoch_step is None:
            epoch_step = int(progress[:progress.find('/')])

        if train_roc_auc in current_metrics:
            metrics[config_file]['train'].append(float(current_metrics[train_roc_auc]))
        elif validation_roc_auc in current_metrics:
            metrics[config_file]['validation'].append(float(current_metrics[validation_roc_auc]))
        elif test_roc_auc in current_metrics:
            metrics[config_file]['test'].append(float(current_metrics[test_roc_auc]))

print('\nFound metrics:')
print(metrics)

print(f'\nWriting plots to {output_path}...')
for key in tqdm(metrics.keys()):
    train_metrics = metrics[key]['train']
    plt.plot([(i + 1) * epoch_step for i in range(len(train_metrics))], train_metrics)
    plt.title(f'ROC AUC for {key}')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.savefig(os.path.join(output_path, f'{key.replace("/", "-")}-train-roc-auc.png'))
    plt.clf()
