import os
import random
import shutil
import argparse
from typing import List, Optional

from tqdm import tqdm
from loguru import logger

from utils import get_images, load_yaml


def check_dir(
    dir_path: str,
    clean: Optional[bool] = False
) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f'MakeDir: {dir_path}')
    else:
        if clean:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            logger.info(f'Clean and MakeDir: {dir_path}')


def copy_image(
    images: List[str],
    output: str,
    mode: Optional[str] = 'copy'
) -> None:
    for image in tqdm(images):
        if mode == 'copy':
            shutil.copy(image, output)
        elif mode == 'move':
            shutil.move(image, output)


def split_data(
    images: List[str],
    ratio: List[float],
    output_paths: List[str]
) -> None:
    num_of_images = len(images)
    if num_of_images == 0:
        return
    random.shuffle(images)

    train_offset = int(num_of_images * ratio[0])
    val_offset = int(num_of_images * ratio[1])
    # test_offset = int(num_of_images * ratio[2])

    train_sublist = images[:train_offset]
    val_sublist = images[train_offset:train_offset + val_offset]
    test_sublist = images[train_offset + val_offset:]
    train_path, val_path, test_path = output_paths[0], output_paths[1], output_paths[2],
    for sublist, output_path in zip(
        [train_sublist, val_sublist, test_sublist],
        [train_path, val_path, test_path]
    ):
        copy_image(sublist, output_path)


def main(
    root: str,
    output: Optional[str] = None,
    ratio: List[float] = None,
    labels: Optional[List[str]] = None,
    clear_output: Optional[bool] = True
) -> None:
    if not os.path.exists(root):
        logger.error(f'Path: {root} is not exists.')
        return

    if os.path.exists(output):
        if clear_output:
            shutil.rmtree(output)
            logger.info(f'Clear output dir: {output}.')
        else:
            logger.error(f'Output: {output} dir is exists.')
            return

    if ratio is None:
        logger.error('ratio is empty.')
        # ratio = [0.6, 0.2, 0.2]
    if labels is None:
        labels = []

    root_dir, _ = os.path.split(root)

    train_path: str
    val_path: str
    test_path: str

    if len(labels) == 0:
        if output is not None:
            train_path = os.path.join(output, 'train')
            val_path = os.path.join(output, 'val')
            test_path = os.path.join(output, 'test')
        else:
            train_path = os.path.join(root_dir, 'train')
            val_path = os.path.join(root_dir, 'val')
            test_path = os.path.join(root_dir, 'test')

        for path in [train_path, val_path, test_path]:
            check_dir(path)

        images = get_images(root)
        split_data(images, ratio, [train_path, val_path, test_path])

    else:
        for label in labels:
            label_data_path = os.path.join(root, label)
            if os.path.exists(label_data_path) is False:
                logger.error(f'Can`t found dir: {label_data_path}')
                exit()

            if output is not None:
                train_path = os.path.join(output, 'train', label)
                val_path = os.path.join(output, 'val', label)
                test_path = os.path.join(output, 'test', label)
            else:
                train_path = os.path.join(root_dir, 'train', label)
                val_path = os.path.join(root_dir, 'val', label)
                test_path = os.path.join(root_dir, 'test', label)

            for path in [train_path, val_path, test_path]:
                check_dir(path)

            images = get_images(label_data_path)
            split_data(images, ratio, [train_path, val_path, test_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=r'',
        help='CONFIG path'
    )
    parser.add_argument(
        '--root', type=str, default=r'',
        help='Root path.',
    )
    parser.add_argument(
        '--output', type=str, default=r'',
        help='Output path.',
    )
    parser.add_argument(
        '--ratio', type=list, default=[],
        help='Train Val Test ratios,i.e.[0.6, 0.2, 0.2]',
    )
    parser.add_argument(
        '--labels', type=list, default=r'',
        help='i.e.[label1,label2,label3]',
    )
    parser.add_argument(
        '--clear', type=bool, default=True,
        help='i.e.[label1,label2,label3]',
    )
    args = parser.parse_args()

    config: dict
    if args.config is not None:
        config = load_yaml(args.config)
    else:
        config = {
            "root": args.root,
            "input": args.input,
            "output": args.output,
            "ratio": args.ratio,
            "labels": args.labels,
            "clear_output": args.clear_output
        }

    main(
        config['root'],
        config['output'],
        config['ratio'],
        config['labels'],
        config['clear_output']
    )
