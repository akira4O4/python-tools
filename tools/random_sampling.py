import os
from random import sample
from utils import get_images
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    random_rate = 128
    mode = 'copy'
    root = r'D:\llf\dataset\dog_cat\all'
    output = r'D:\llf\dataset\dog_cat\128'

    images = get_images(root)
    images_len = len(images)
    if random_rate < 1:
        new_data = sample(images, int(images_len * random_rate))
    else:
        new_data = sample(images, random_rate)

    for image in tqdm(new_data):

        image_basename = os.path.basename(image)
        new_image_path = image.replace(root, output)
        new_image_dir = new_image_path.split(image_basename)[0]

        if not os.path.exists(new_image_dir):
            os.makedirs(new_image_dir, exist_ok=True)
        if mode.lower() == 'copy':
            shutil.copy(image, new_image_dir)
        elif mode.lower() == 'move':
            shutil.move(image, new_image_dir)
