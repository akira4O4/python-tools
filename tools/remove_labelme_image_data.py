import os
from utils import load_json, save_json, get_images
from tqdm import tqdm

if __name__ == '__main__':
    root = r'D:\llf\dataset\danyang\training_data\D2\train\20240724\png'
    for image in tqdm(get_images(root)):
        basename = os.path.basename(image)
        name, suffix = os.path.splitext(basename)
        json_path = image.replace(suffix, '.json')
        if os.path.exists(json_path):
            json_data = load_json(json_path)
            json_data['imageData'] = None
            save_json(json_data, json_path)
