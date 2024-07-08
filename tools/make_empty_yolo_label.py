import os
from utils import get_images
from tqdm import tqdm

if __name__ == '__main__':
    image_path = r''
    images = get_images(image_path)
    for image in tqdm(images):
        basenaem = os.path.basename(image)
        name, suffix = os.path.splitext(basenaem)
        label = basenaem.replace(suffix, '.txt')
        label_path = os.path.join(image_path, label)
        with open(label_path, 'w') as f:
            f.writelines('')
