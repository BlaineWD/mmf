# Image Augmentation

Script to augment the hateful memes dataset

Create conda environment: `conda env create -f requirements.yml`

Activate conda environment: `conda activate image_augmentation`

Run these from project root (up a level):

Run script and replace all images with randomly augmented versions of images: `python image_augmentation/randomly_augment_images.py`

Run script and replace 50% of images with randomly augmented versions of images: `python image_augmentation/randomly_augment_images.py --replace_ratio 0.5`

You can also use `--input_path` and `--output_path` to set image folder paths
