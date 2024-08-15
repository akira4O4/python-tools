import os
import gzip

import cv2
import numpy as np
from tqdm import tqdm


def load_mnist(path: str, kind: str = 'train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


if __name__ == '__main__':
    path = r'C:\Users\Lee Linfeng\Desktop\fashion\sources'
    output = r'C:\Users\Lee Linfeng\Desktop\fashion\images'
    kind = 'train'
    class_names = [
        'T-shirt',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle_boot'
    ]
    images, labels = load_mnist(path, kind)
    for i in tqdm(range(images.shape[0])):
        image = images[i].reshape(28, 28)
        label = labels[i]
        save_dir = os.path.join(output, kind, class_names[int(label)])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(save_dir, f'{kind}_{i}.png'), image)
