import cv2
import albumentations as A
import os
import argparse
import random
import shutil
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='hateful_memes', help='Input path of unaltered hateful memes dataset')
parser.add_argument('--output_path', type=str, default='hateful_memes_augmented', help='Output path of augmented (altered) hateful memes dataset')
parser.add_argument('--replace_ratio', type=float, help='What proportion of hateful_memes images to replace with randomly augmented versions of images')

args = parser.parse_args()
replace_ratio = args.replace_ratio
input_path = args.input_path
output_path = args.output_path

if not os.path.exists(input_path):
    print(f'Input path {input_path} does not exist, please make sure you unzipped the original hateful memes to this path')
    sys.exit()

print(f'Creating augmented dataset from {input_path} and saving to {output_path}...')

if replace_ratio is not None:
    print(f'Replace ratio is {replace_ratio}')

# TODO: Remove some of these or lessen their impact since they are pretty drastic transformations
transform = A.Compose([
    A.CLAHE(),
    A.RandomRotate90(),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                    rotate_limit=45, p=.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue()
])

print('Coping jsonl and other files...')
input_data_path = os.path.join(input_path, 'data')
output_data_path = os.path.join(output_path, 'data')
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)
input_files = os.listdir(input_data_path)
valid_file_names = ['dev_seen.jsonl', 'dev_unseen.jsonl', 'LICENSE.txt', 'README.md', 'test_seen.jsonl', 'test_unseen.jsonl', 'train.jsonl']
for file_name in tqdm(input_files):
    if file_name not in valid_file_names:
        continue
    file_input_path = os.path.join(input_data_path, file_name)
    file_output_path = os.path.join(output_data_path, file_name)
    shutil.copy(file_input_path, file_output_path)

print('Copying images...')
input_images_path = os.path.join(input_data_path, 'img')
output_images_path = os.path.join(output_data_path, 'img')
if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
for _, _, files in os.walk(input_images_path):
    # files_to_augment is an array of file names that we will apply transform to
    # All files in files array not in files_to_augment array will be copied without augmentation
    if replace_ratio is not None:
        files_to_augment = random.sample(files, int(replace_ratio * len(files)))
    else:
        files_to_augment = files.copy()

    for file_name in tqdm(files):
        file_input_path = os.path.join(input_images_path, file_name)
        file_output_path = os.path.join(output_images_path, file_name)

        if file_name in files_to_augment:
            image = cv2.imread(file_input_path)
            augmented_image = transform(image=image)['image']
            cv2.imwrite(file_output_path, augmented_image)
        else:
            shutil.copy(file_input_path, file_output_path)
