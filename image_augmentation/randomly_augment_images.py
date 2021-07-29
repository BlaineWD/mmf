import cv2
import albumentations as A
import matplotlib.pyplot as plt
import os
import argparse
import random
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='hateful_memes/data/img', help='Input path of unaltered hateful memes dataset')
parser.add_argument('--output_path', type=str, default='hateful_memes_augmented/data/img', help='Output path of augmented (altered) hateful memes dataset')
parser.add_argument('--replace_ratio', type=float, help='What proportion of hateful_memes images to replace with randomly augmented versions of images')

args = parser.parse_args()
replace_ratio = args.replace_ratio
input_path = args.input_path
output_path = args.output_path

print(f'Replace ratio is {replace_ratio}')

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

if not os.path.exists(output_path):
    os.makedirs(output_path)

print(f'Creating augmented dataset from {input_path} and saving to {output_path}...')
for _, _, files in os.walk(input_path):
    files = files[:100]
    # Files to augment is an array of file names that we will apply transform to
    # All files in files array not in files_to_augment array will be copied without augmentation
    if replace_ratio is not None:
        files_to_augment = random.sample(files, replace_ratio * len(files))
    else:
        files_to_augment = files.copy()

    for file_name in tqdm(files):
        file_input_path = os.path.join(input_path, file_name)
        file_output_path = os.path.join(output_path, file_name)

        if file_name in files_to_augment:
            image = cv2.imread(file_input_path)
            augmented_image = transform(image=image)['image']
            cv2.imwrite(file_output_path, augmented_image)
        else:
            shutil.copy(file_input_path, file_output_path)
