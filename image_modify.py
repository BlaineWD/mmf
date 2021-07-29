import cv2
import albumentations as A
import matplotlib.pyplot as plt
import os

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

image_input_path = '../../Downloads/hateful_memes/data/img'
augmented_image_output_path = 'hateful_memes_augmented/data/img'
if not os.path.exists(augmented_image_output_path):
    os.makedirs(augmented_image_output_path)

DEV_LIMIT = 10
for _, _, files in os.walk(image_input_path):
    for file_name in files:
        image = cv2.imread(os.path.join(image_input_path, file_name))

        augmented_image = transform(image=image)['image']

        cv2.imwrite(os.path.join(augmented_image_output_path, file_name), augmented_image)
