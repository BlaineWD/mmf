import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm


output_path = 'experiment_plots'

if not os.path.exists(output_path):
    os.makedirs(output_path)

cross_entropy_key = 'hateful_memes/cross_entropy'
train_cross_entropy = f'train/{cross_entropy_key}'
validation_cross_entropy = f'val/{cross_entropy_key}'
test_cross_entropy = f'test/{cross_entropy_key}'
logistics_line_start = 'mmf.trainers.callbacks.logistics : {'


def get_metrics_from_logs(log_paths):
    metrics = {}
    train_epoch_step = None
    validation_epoch_step = None
    print(f'Reading in experiment logs...')
    for log_file in tqdm(log_paths):
        logistics_lines = []
        with open(log_file) as f:
            log_lines = f.readlines()
            metrics[log_file] = {
                'train_cross_entropy': [], 'validation_cross_entropy': [], 'test_cross_entropy': []
            }

            for line in log_lines[2:]:
                if 'mmf.trainers.callbacks.logistics' in line:
                    logistics_lines.append(line)

        for logistics_line in logistics_lines:
            if 'Finished run' in logistics_line:
                continue

            current_metrics_string = logistics_line[
                                     logistics_line.find(logistics_line_start) + len(logistics_line_start) - 1:]
            current_metrics = json.loads(current_metrics_string)
            progress = current_metrics['progress']

            if train_cross_entropy in current_metrics:
                if train_epoch_step is None:
                    train_epoch_step = int(progress[:progress.find('/')])
                metrics[log_file]['train_cross_entropy'].append(float(current_metrics[train_cross_entropy]))
            elif validation_cross_entropy in current_metrics:
                if validation_epoch_step is None:
                    validation_epoch_step = int(progress[:progress.find('/')])
                metrics[log_file]['validation_cross_entropy'].append(
                    float(current_metrics[validation_cross_entropy]))
            elif test_cross_entropy in current_metrics:
                metrics[log_file]['test_cross_entropy'].append(float(current_metrics[test_cross_entropy]))

    print('\nFound metrics:')
    print(metrics)
    print(f'\nTrain step: {train_epoch_step}')
    print(f'\nValidation step: {validation_epoch_step}')
    return metrics, train_epoch_step, validation_epoch_step


def write_plots(metrics, metric_type, model_name, epoch_step, output_path):
    figure(figsize=(10, 8))
    x_axis_name = 'Iteration'
    print(f'\nWriting {metric_type} plots to {output_path}...')
    metric_config_names = metrics.keys()

    plt.title(f'{metric_type.title()} Cross Entropy Loss across Augmentation levels for {model_name}')
    plt.xlabel(x_axis_name)
    plt.ylabel('Cross Entropy Loss')

    for key in tqdm(metric_config_names):
        cross_entropy_metrics = metrics[key][f'{metric_type}_cross_entropy']
        epoch_increments = [(i + 1) * epoch_step for i in range(len(cross_entropy_metrics))]
        plt.plot(epoch_increments, cross_entropy_metrics)

    augmentation_levels = [
        "unmodified",
        "25% of images transformed",
        "50% of images transformed",
        "75% of images transformed",
        "100% of images transformed"
    ]
    plt.legend(augmentation_levels, ncol=2, loc='best')
    plt.savefig(os.path.join(output_path, f'{metric_type}-{model_name.replace(" ", "-").lower()}-cross-entropy.png'))
    plt.clf()

visual_bert_input_paths = [
    os.path.join('experiment_logs_unaugmented', 'train_2021_07_30T06_16_25.log'),
    os.path.join('experiment_logs_0.25_augmented', 'train_2021_08_01T07_29_23.log'),
    os.path.join('experiment_logs_0.5_augmented', 'train_2021_08_01T02_07_12.log'),
    os.path.join('experiment_logs_0.75_augmented', 'train_2021_08_01T16_22_49.log'),
    os.path.join('experiment_logs_augmented', 'train_2021_07_31T13_23_55.log')
]

metrics, train_epoch_step, validation_epoch_step = get_metrics_from_logs(visual_bert_input_paths)
write_plots(metrics, 'train', 'Visual BERT', train_epoch_step, output_path)
write_plots(metrics, 'validation', 'Visual BERT', validation_epoch_step, output_path)

visual_bert_with_coco_input_paths = [
    os.path.join('experiment_logs_unaugmented', 'train_2021_07_30T07_07_02.log'),
    os.path.join('experiment_logs_0.25_augmented', 'train_2021_08_01T08_44_50.log'),
    os.path.join('experiment_logs_0.5_augmented', 'train_2021_08_01T03_22_25.log'),
    os.path.join('experiment_logs_0.75_augmented', 'train_2021_08_01T17_38_07.log'),
    os.path.join('experiment_logs_augmented', 'train_2021_07_31T14_39_11.log')
]

metrics, train_epoch_step, validation_epoch_step = get_metrics_from_logs(visual_bert_with_coco_input_paths)
write_plots(metrics, 'train', 'Visual BERT COCO', train_epoch_step, output_path)
write_plots(metrics, 'validation', 'Visual BERT COCO', validation_epoch_step, output_path)
