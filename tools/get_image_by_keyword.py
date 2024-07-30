import argparse
import os
import shutil
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


def main(root: str, output: str, keyword: str) -> None:
    if os.path.exists(output) is False:
        os.makedirs(output)

    for image in tqdm(get_images(root)):
        basename = os.path.basename(image)
        if keyword in basename:
            shutil.copy(image, os.path.join(output, basename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the image by keyword.')

    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--keyword', type=str)
    args = parser.parse_args()

    main(args.root, args.output, args.keyword)
