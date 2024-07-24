import os
import cv2
import json
from tqdm import tqdm


def get_images(path: str, ext=None) -> list:
    if ext is None:
        ext = ['.png', '.jpg', '.bmp']
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext in ext:
                image = os.path.join(root, file)
                data.append(image)
    return data


def load_json(path: str):
    with open(path, 'r') as config_file:
        data = json.load(config_file)  # 配置字典
    return data


def save_json(data, save_path: str, sort=False) -> None:
    with open(save_path, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=sort))


def main(root: str, output: str, new_suffix: str) -> None:
    if os.path.exists(output) is False:
        os.makedirs(output)

    for image in tqdm(get_images(root)):
        im = cv2.imread(image)

        basename = os.path.basename(image)
        name, suffix = os.path.splitext(basename)

        new_image_path = os.path.join(output, name) + new_suffix
        cv2.imwrite(new_image_path, im)

        json_file = image.replace(suffix, '.json')
        if os.path.exists(json_file):
            json_data = load_json(json_file)
            json_data['imagePath'] = name + new_suffix
            new_json_path = os.path.join(output, name) + '.json'
            save_json(json_data, new_json_path)


if __name__ == '__main__':
    root = r'D:\llf\dataset\danyang\training_data\D2\train\bakcground.1\jpg'
    output = r'D:\llf\dataset\danyang\training_data\D2\train\bakcground.1\png'
    new_suffix = '.png'

    main(root, output, new_suffix)
