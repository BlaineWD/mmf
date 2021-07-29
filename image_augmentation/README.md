# Image Augmentation

Script to augment the hateful memes dataset

Create conda environment: `conda env create -f requirements.yml`

Activate conda environment: `conda activate image_augmentation`

Run script and replace all images with randomly augmented versions of images: `python randomly_augment_images.py`

Run script and replace 50% of images with randomly augmented versions of images: `python randomly_augment_images.py --replace_ratio 0.5`
