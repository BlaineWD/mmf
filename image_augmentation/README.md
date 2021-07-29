# Image Augmentation

Script to augment the hateful memes dataset

## Conda

Create conda environment: `conda env create -f requirements.yml`

Activate conda environment: `conda activate image_augmentation`

## Setup

1. Download the unmodified hateful_memes.zip
2. Unzip it to project root (up a folder from this README).  Check that it is a folder called `hateful_memes` with a `data` folder inside. Example: `unzip -q -d hateful_memes ateful_memes.zip`
3. Make sure that `hateful_memes_augmented` folder does not exist or is empty

## Running

Run these from project root:

Run script and replace all images with randomly augmented versions of images: `python image_augmentation/randomly_augment_images.py`

Run script and replace 50% of images with randomly augmented versions of images: `python image_augmentation/randomly_augment_images.py --replace_ratio 0.5`

You can also use `--input_path` and `--output_path` to set image folder paths

## After 

You can then zip up `hateful_memes_augmented/data` into a `.zip` to be used by `mmf_config_hm` CLI
