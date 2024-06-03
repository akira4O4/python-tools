from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger


class Transform:
    def __init__(self, ops: dict) -> None:
        self._ops = ops
        self._register = {
            'ToNumpy': self.to_numpy,
            'ToPIL': self.to_pil,
            'GRAY2BGR': self.gray2bgr,
            'GRAY2RGB': self.gray2rgb,
            'RGB2BGR': self.rgb2bgr,
            'BGR2RGB': self.bgr2rgb,
            'Resize': self.resize,
            'HWC2CHW': self.hwc2chw,
            'CHW2HWC': self.chw2hwc,
            'Normalize': self.normalize,
            'Standardize': self.standardize,
            'Expand_Dims': self.expand_dims,
            'NCHW2CHW': self.nchw2chw
        }

    def add_op(self, ops: dict) -> None:
        self._ops.update(ops)

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        for name, args in self._ops.items():
            args = {} if args is None else args
            img = self._register[name](img, **args)
        return np.ascontiguousarray(img)

    @staticmethod
    def to_numpy(img: Image.Image) -> Optional[np.ndarray]:
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # noqa
            return img
        else:
            logger.error(f'Input type must be PIL.Image.Image,Your input type:{type(img)}.')
            return None

    @staticmethod
    def to_pil(img: np.ndarray) -> Optional[Image.Image]:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return img
        else:
            logger.error(f'Input type must be np.ndarray,Your input type:{type(img)}.')
            return None

    @staticmethod
    def gray2bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def gray2rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def rgb2bgr(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def bgr2rgb(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def resize(img: np.ndarray, wh: List[int]) -> np.ndarray:
        h, w, c = img.shape
        if h != wh[1] or w != wh[0]:
            img = cv2.resize(img, tuple(wh))
        return img

    @staticmethod
    def hwc2chw(img: np.ndarray) -> np.ndarray:
        img = img.transpose((2, 0, 1))
        return img

    @staticmethod
    def chw2hwc(img: np.ndarray) -> np.ndarray:
        img = img.transpose((1, 2, 0))
        return img

    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        return img / 255.0

    @staticmethod
    def standardize(img: np.ndarray) -> np.ndarray:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img -= np.array(mean).reshape((3, 1, 1))
        img /= np.array(std).reshape((3, 1, 1))
        return img

    # CHW->1CHW
    @staticmethod
    def expand_dims(img: np.ndarray) -> np.ndarray:
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    @staticmethod
    def nchw2chw(img: np.ndarray) -> Optional[np.ndarray]:  # noqa
        if len(img.shape) != 4:
            logger.error(f'Input shape must be 4dims,Your input shape:{img.shape}')
            return None
        if img.shape[0] == 1:
            return img.squeeze()
        else:
            return img[0, :, :]

    @staticmethod
    def concatenate(images: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(tuple(images), axis=0)


def easy_preprocess(img: np.ndarray, wh: List[int]) -> np.ndarray:
    image_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(image_data, (wh[0], wh[1]))
    image_data = np.array(image_data) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))

    image_data -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    image_data /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data
