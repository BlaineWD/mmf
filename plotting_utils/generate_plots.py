import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
cross_entropy_key = 'hateful_memes/cross_entropy'
train_cross_entropy = f'train/{cross_entropy_key}'
validation_cross_entropy = f'val/{cross_entropy_key}'
test_cross_entropy = f'test/{cross_entropy_key}'
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
        metrics[config_file] = {
            'train_roc': [], 'validation_roc': [], 'test_roc': [],
            'train_cross_entropy': [], 'validation_cross_entropy': [], 'test_cross_entropy': []
        }

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
            metrics[config_file]['train_roc'].append(float(current_metrics[train_roc_auc]))
        elif validation_roc_auc in current_metrics:
            metrics[config_file]['validation_roc'].append(float(current_metrics[validation_roc_auc]))
        elif test_roc_auc in current_metrics:
            metrics[config_file]['test_roc'].append(float(current_metrics[test_roc_auc]))

        if train_cross_entropy in current_metrics:
            metrics[config_file]['train_cross_entropy'].append(float(current_metrics[train_cross_entropy]))
        elif validation_cross_entropy in current_metrics:
            metrics[config_file]['validation_cross_entropy'].append(float(current_metrics[validation_cross_entropy]))
        elif test_cross_entropy in current_metrics:
            metrics[config_file]['test_cross_entropy'].append(float(current_metrics[test_cross_entropy]))

print('\nFound metrics:')
print(metrics)

print(f'\nWriting plots to {output_path}...')
figure(figsize=(8, 6))
for key in tqdm(metrics.keys()):
    train_roc_metrics = metrics[key]['train_roc']
    epoch_increments = [(i + 1) * epoch_step for i in range(len(train_roc_metrics))]
    plt.plot(epoch_increments, train_roc_metrics)
    plt.title(f'ROC AUC for {key}')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.savefig(os.path.join(output_path, f'{key.replace("/", "-")}-train-roc-auc.png'))
    plt.clf()

    train_cross_entropy = metrics[key]['train_cross_entropy']
    epoch_increments = [(i + 1) * epoch_step for i in range(len(train_cross_entropy))]
    plt.plot(epoch_increments, train_cross_entropy)
    plt.title(f'Cross Entropy Loss for {key}')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.savefig(os.path.join(output_path, f'{key.replace("/", "-")}-train-cross-entropy.png'))
    plt.clf()

# TODO: Plot validation and test loss?
