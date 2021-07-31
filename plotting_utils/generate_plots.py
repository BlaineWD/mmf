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
accuracy_key = 'hateful_memes/accuracy'
train_accuracy = f'train/{accuracy_key}'
validation_accuracy = f'val/{accuracy_key}'
test_accuracy = f'test/{accuracy_key}'
logistics_line_start = 'mmf.trainers.callbacks.logistics : {'

log_files = os.listdir(input_path)
metrics = {}
train_epoch_step = None
validation_epoch_step = None
print(f'Reading in experiment logs from {input_path}...')
for log_file in tqdm(log_files):
    logistics_lines = []
    with open(os.path.join(input_path, log_file)) as f:
        log_lines = f.readlines()
        model_description_line = log_lines[1]
        config_file = model_description_line[model_description_line.find('config=') + 7:model_description_line.find("', 'model")]
        metrics[config_file] = {
            'train_roc': [], 'validation_roc': [], 'test_roc': [],
            'train_cross_entropy': [], 'validation_cross_entropy': [], 'test_cross_entropy': [],
            'train_accuracy': [], 'validation_accuracy': [], 'test_accuracy': []
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

        if train_roc_auc in current_metrics:
            if train_epoch_step is None:
                train_epoch_step = int(progress[:progress.find('/')])
            metrics[config_file]['train_roc'].append(float(current_metrics[train_roc_auc]))
        elif validation_roc_auc in current_metrics:
            if validation_epoch_step is None:
                validation_epoch_step = int(progress[:progress.find('/')])
            metrics[config_file]['validation_roc'].append(float(current_metrics[validation_roc_auc]))
        elif test_roc_auc in current_metrics:
            metrics[config_file]['test_roc'].append(float(current_metrics[test_roc_auc]))

        if train_cross_entropy in current_metrics:
            metrics[config_file]['train_cross_entropy'].append(float(current_metrics[train_cross_entropy]))
        elif validation_cross_entropy in current_metrics:
            metrics[config_file]['validation_cross_entropy'].append(float(current_metrics[validation_cross_entropy]))
        elif test_cross_entropy in current_metrics:
            metrics[config_file]['test_cross_entropy'].append(float(current_metrics[test_cross_entropy]))

        if train_accuracy in current_metrics:
            metrics[config_file]['train_accuracy'].append(float(current_metrics[train_accuracy]))
        elif validation_accuracy in current_metrics:
            metrics[config_file]['validation_accuracy'].append(float(current_metrics[validation_accuracy]))
        elif test_accuracy in current_metrics:
            metrics[config_file]['test_accuracy'].append(float(current_metrics[test_accuracy]))

print('\nFound metrics:')
print(metrics)

config_to_baseline_name_mapping = {
    'projects/hateful_memes/configs/unimodal/image.yaml': 'Image-Grid',
    'projects/hateful_memes/configs/unimodal/with_features.yaml': 'Image-Region',
    'projects/hateful_memes/configs/unimodal/bert.yaml': 'Text BERT',
    'projects/hateful_memes/configs/late_fusion/defaults.yaml': 'Late Fusion',
    'projects/hateful_memes/configs/mmbt/defaults.yaml': 'MMBT-Grid',
    'projects/hateful_memes/configs/mmbt/with_features.yaml': 'MMBT-Region',
    'projects/hateful_memes/configs/vilbert/defaults.yaml': 'ViLBERT',
    'projects/hateful_memes/configs/visual_bert/direct.yaml': 'Visual BERT',
    'projects/hateful_memes/configs/vilbert/from_cc.yaml': 'ViLBERT CC',
    'projects/hateful_memes/configs/visual_bert/from_coco.yaml': 'Visual BERT COCO'
}


def write_plots(metrics, metric_type, epoch_step, output_path):
    figure(figsize=(8, 6))
    x_axis_name = 'Iteration'
    print(f'\nWriting {metric_type} plots to {output_path}...')
    metric_config_names = metrics.keys()
    baseline_names = []
    for key in tqdm(metric_config_names):
        baseline_name = config_to_baseline_name_mapping[key]
        baseline_names.append(baseline_name)

    plt.title(f'{metric_type.title()} ROC AUC over baseline models')
    plt.xlabel(x_axis_name)
    plt.ylabel('ROC AUC')

    for key in tqdm(metric_config_names):
        roc_metrics = metrics[key][f'{metric_type}_roc']
        epoch_increments = [(i + 1) * epoch_step for i in range(len(roc_metrics))]
        plt.plot(epoch_increments, roc_metrics)

    plt.legend(baseline_names)
    plt.savefig(os.path.join(output_path, f'{metric_type}-roc-auc.png'))
    plt.clf()

    plt.title(f'{metric_type.title()} Cross Entropy Loss over baseline models')
    plt.xlabel(x_axis_name)
    plt.ylabel('Cross Entropy Loss')

    for key in tqdm(metric_config_names):
        cross_entropy_metrics = metrics[key][f'{metric_type}_cross_entropy']
        epoch_increments = [(i + 1) * epoch_step for i in range(len(cross_entropy_metrics))]
        plt.plot(epoch_increments, cross_entropy_metrics)

    plt.legend(baseline_names)
    plt.savefig(os.path.join(output_path, f'{metric_type}-cross-entropy.png'))
    plt.clf()

    plt.title(f'{metric_type.title()} Accuracy over baseline models')
    plt.xlabel(x_axis_name)
    plt.ylabel('Accuracy')

    for key in tqdm(metric_config_names):
        accuracy_metrics = metrics[key][f'{metric_type}_accuracy']
        epoch_increments = [(i + 1) * epoch_step for i in range(len(accuracy_metrics))]
        plt.plot(epoch_increments, accuracy_metrics)

    plt.legend(baseline_names)
    plt.savefig(os.path.join(output_path, f'{metric_type}-accuracy.png'))
    plt.clf()


write_plots(metrics, 'train', train_epoch_step, output_path)
write_plots(metrics, 'validation', validation_epoch_step, output_path)

# TODO: Figure out what other metrics we want to add
