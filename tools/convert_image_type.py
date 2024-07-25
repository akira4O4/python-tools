import os
import cv2

from tqdm import tqdm
from utils import get_images, save_json, load_json, imread, imwrite


def main(root: str, output: str, new_suffix: str) -> None:
    if os.path.exists(output) is False:
        os.makedirs(output)

    for image in tqdm(get_images(root)):
        im = imread(image)
        basename = os.path.basename(image)
        name, suffix = os.path.splitext(basename)

        new_image_path = os.path.join(output, name) + new_suffix
        imwrite(new_image_path, im, new_suffix)

        json_file = image.replace(suffix, '.json')
        if os.path.exists(json_file):
            json_data = load_json(json_file)
            json_data['imagePath'] = name + new_suffix
            new_json_path = os.path.join(output, name) + '.json'
            save_json(json_data, new_json_path)


if __name__ == '__main__':
    root = r'C:\Users\Lee Linfeng\Desktop\temp\你好'
    output = r'C:\Users\Lee Linfeng\Desktop\temp\output'
    new_suffix = '.png'

    main(root, output, new_suffix)
