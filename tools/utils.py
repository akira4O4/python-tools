import os
import time
import json
import shutil
from typing import Union, Optional

import cv2
import numpy as np
import yaml
from loguru import logger

__all__ = [
    'letterbox',
    'check_exists',
    'check_dir',
    'load_json',
    'load_yaml',
    'get_time',
    'get_images',
    'timer',
]
COLOR_LIST = np.random.uniform(0, 255, size=(80, 3))

color_list = [
    [0, 0, 0],
    [0, 255, 0], [0, 0, 255], [0, 255, 255],
    [255, 255, 0], [0, 255, 255], [255, 255, 0],
    [255, 255, 255], [170, 255, 255],
    [255, 0, 170], [85, 0, 255], [128, 255, 128],
    [170, 255, 255], [0, 255, 170], [85, 0, 255],
    [170, 0, 255], [0, 85, 255], [0, 170, 255],
    [255, 255, 85], [255, 255, 170], [255, 0, 255],
    [255, 85, 255], [255, 170, 255], [85, 255, 255],
]


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f'🕐 {func.__name__} Spend Time: {format(time_spend, ".3f")}s')
        return result

    return func_wrapper


def letterbox(
        image_src: np.ndarray,
        dst_size: Union[tuple, list],
        pad_color: Optional[Union[tuple, list]] = (114, 114, 114)
) -> tuple:
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image_src.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(image_src, (pad_w, pad_h),
                               interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = image_src

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    image_dst = cv2.copyMakeBorder(
        image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
    return image_dst, x_offset, y_offset


def check_exists(path) -> bool:
    if not os.path.exists(path):
        logger.info(f'{path} is not found.')
        return False
    return True


def check_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def round3(x: Union[float, np.float64]) -> float:
    if isinstance(x, float):
        return round(x, 3)
    elif isinstance(x, np.float64) or isinstance(x, np.float32):
        return float(np.round(x, 3))


def save_json(data: Union[dict, list], output: str, sort: Optional[bool] = False) -> None:
    with open(output, 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, sort_keys=sort))
    logger.info(f'Save Json File in: {output}.')


def save_yaml(data: dict, output: str, sort: Optional[bool] = False) -> None:
    with open(output, 'a', encoding='utf-8') as f:
        yaml.dump(data=data, stream=f, allow_unicode=True, sort_keys=sort)
    logger.info(f'Save Yaml File in: {output}.')


def load_json(path: str):
    with open(path, 'r') as config_file:
        data = json.load(config_file)  # 配置字典
    return data


def load_yaml(path: str):
    with open(path, encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def get_time(fmt: str = '%Y%m%d_%H%M%S') -> str:
    time_str = time.strftime(fmt, time.localtime())
    # return '2022 - 11 - 14 - (17:35:42)'
    return str(time_str)


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
