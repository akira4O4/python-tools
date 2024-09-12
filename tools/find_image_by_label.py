import os
from tqdm import tqdm
import shutil

from utils import get_images, load_json, check_dir

if __name__ == '__main__':
    count = 0
    label = '1_line'
    input_dir = r'D:\llf\dataset\danyang\新视客\train\D2\train'
    output_dir = r'D:\llf\dataset\danyang\新视客\train\D2\output'

    check_dir(output_dir)
    for image in tqdm(get_images(input_dir)):
        basename = os.path.basename(image)
        name, suffix = os.path.splitext(basename)
        json_file = image.replace(suffix, '.json')
        if os.path.exists(json_file):
            json_data = load_json(json_file)
            shapes = json_data['shapes']
            for shape in shapes:
                if shape['label'] == label:
                    shutil.move(json_file, output_dir)
                    shutil.move(image, output_dir)
                    count += 1
                    break
    print(f'Move {count} data.')
