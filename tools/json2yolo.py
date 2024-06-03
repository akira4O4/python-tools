from typing import Optional, List
import json
import os
import numpy as np
from utils import load_json, get_images
from tqdm import tqdm


def find_labels(root: str) -> list:
    labels = []

    images = get_images(root)
    for image in images:
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)
        json_file_path = os.path.join(root, basename.replace(ext, '.json'))

        if not os.path.exists(json_file_path):
            continue

        json_data = load_json(json_file_path)
        # "shapes": [
        #     {
        #         "label": "qipao",
        #         "points": []
        #       }
        # ]
        for shape in json_data.get("shapes"):
            if shape["label"] not in labels:
                labels.append(shape["label"])

    return sorted(labels)


def xywh2yolo(box: list, image_wh: list) -> List[int]:
    w, h = image_wh[0], image_wh[1]
    box = np.array(box, dtype=np.float64)
    box[:2] += box[2:] / 2
    box[[0, 2]] /= w
    box[[1, 3]] /= h
    return box.tolist()


def xyxy2yolo(xyxy: list, image_wh: list) -> List[int]:
    w, h = image_wh[0], image_wh[1]
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bbox_w = abs(x2 - x1) / w
    bbox_h = abs(y2 - y1) / h
    return [cx, cy, bbox_w, bbox_h]


def cxcywh2xyxy(data: list, image_wh: list) -> List[int]:
    w, h = image_wh[0], image_wh[1]
    x_, y_, w_, h_ = data[0], data[1], data[2], data[3]
    x1 = w * x_ - 0.5 * w * w_
    y1 = h * y_ - 0.5 * h * h_

    x2 = w * x_ + 0.5 * w * w_
    y2 = h * y_ + 0.5 * h * h_
    return [x1, y1, x2, y2]


def main(root: str, output: Optional[str] = None, labels=None, label_type: str = 'xyxy') -> None:
    if labels is None:
        labels = []

    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
            print(f'Make dir: {output}')
    else:
        output = root

    images = get_images(root)
    for image in tqdm(images):
        basename = os.path.basename(image)
        name, ext = os.path.splitext(basename)

        json_file_path = os.path.join(root, basename.replace(ext, '.json'))
        txt_file_path = os.path.join(output, basename.replace(ext, '.txt'))
        txt_file = open(txt_file_path, 'w')

        if not os.path.exists(json_file_path):
            continue

        json_data = load_json(json_file_path)
        image_w = json_data['imageWidth']
        image_h = json_data['imageHeight']

        for i in range(len(json_data['shapes'])):
            label = json_data['shapes'][i]['label']

            index = labels.index(label)

            x1 = json_data['shapes'][i]['points'][0][0]
            x2 = json_data['shapes'][i]['points'][1][0]

            y1 = json_data['shapes'][i]['points'][0][1]
            y2 = json_data['shapes'][i]['points'][1][1]

            # min_x = min(x1, x2)
            # min_y = min(y1, y2)
            # max_x = max(x1, x2)
            # max_y = max(y1, y2)

            # data = [min_x, min_y, max_x, max_y]
            data = [x1, y1, x2, y2]
            print(f'src:{data}')

            bbox = []
            if label_type == 'xyxy':
                bbox = xyxy2yolo(data, [image_w, image_h])
            elif label_type == 'xywh':
                bbox = xywh2yolo(data, [image_w, image_h])

            temp = cxcywh2xyxy(bbox, [image_w, image_h])
            print(f're: {temp}')
            txt_file.write(str(index) + " " + " ".join([str(a) for a in bbox]) + '\n')
        txt_file.close()


if __name__ == "__main__":
    root = r'D:\llf\dataset\danyang\B\dataset\train\img-crop'
    labels = find_labels(root)
    print(f'labels: {labels}')
    main(root, None, labels)
