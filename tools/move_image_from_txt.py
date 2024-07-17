import os
import shutil

from tqdm import tqdm

from utils import get_files

if __name__ == '__main__':
    root = ''
    image_dir = ''
    output = ''
    files = get_files(root, '.txt')
    for file in tqdm(files):
        basename = os.path.basename(file)
        image_path = os.path.join(image_dir, basename)
        assert os.path.exists(image_path), 'image path is not found.'
        shutil.copy(image_path, output)
