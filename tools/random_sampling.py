import os
from random import sample
from utils import get_images
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    random_rate = 10000
    root = r''
    output = r''

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
        shutil.copy(image, new_image_dir)
