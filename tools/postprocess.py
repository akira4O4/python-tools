import os
import shutil
from copy import deepcopy
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from utils import COLOR_LIST
from utils import check_dir, round3
from box import Location, Box, Detection


def xyxy2xywh(xyxy: List[int]) -> List[int]:  # noqa
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def xywh2xyxy(xywh: List[int]) -> List[int]:  # noqa
    x, y, w, h = xywh
    x2 = x + w
    y2 = y + h
    return [x, y, x2, y2]


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x)
    y = e_x / np.sum(e_x)
    return y


def safe_softmax(x: np.ndarray) -> np.ndarray:
    max_val = np.max(x)
    e_x = np.exp(x - max_val)
    y = e_x / np.sum(e_x)
    return y


def online_softmax(x: np.ndarray) -> np.ndarray:
    old_max = np.array(-float('inf'))
    sum_val = 0

    for item in x:
        new_max = max(old_max, item)
        sum_val = sum_val * np.exp(old_max - new_max) + np.exp(item - new_max)
        old_max = new_max

    output = np.exp(x - old_max) / sum_val
    return output


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-x))


def classification_postprocess(  # noqa
        output: np.ndarray,  # shape:[c,]
        image_path: str,
        labels: List[str],
        thr: List[float],
        save: str,
) -> dict:
    if sum(thr) != 1:
        logger.error('sum(thr)!=1.')
        exit()

    # count = {}
    # for label in labels:
    #     count.update({str(label): []})

    result = {
        "image": image_path,
        "scores": [],
        "thr": 0,
        "score": 0,
        "label": ''
    }

    scores = [round3(x) for x in output]
    result['scores'] = scores
    sort_idx = np.argsort(-np.array(scores))  # down-sort

    for idx in sort_idx:
        label = labels[idx]

        if scores[idx] >= thr[idx]:
            # Copy Image
            copy_dir = os.path.join(save, label)
            check_dir(copy_dir)
            shutil.copy(image_path, copy_dir)

            result['score'] = scores[idx]
            result['label'] = labels[idx]
            result['thr'] = thr[idx]
            break
    return result


def segmentation_postprocessing(
        image_path: str,
        model_output,
        labels: List[str],
        thr: List[int],
        sum_method: Optional[bool] = True
):
    if len(labels) != len(thr):
        logger.error('len(labels) != len(thr)')
        return
    result = {
        "image": image_path,
        "detail": {},
        "gt_thr": []
    }
    predict = model_output[0].copy().astype(np.uint8)  # 0-255

    # model_output.shape=(c,h,w)
    predict_plot = np.zeros((model_output.shape[1], model_output.shape[2], 3)).astype(np.uint8)  # + [0, 0, 0]

    for label_id in range(0, len(labels)):

        if label_id == 0:
            continue

        label = labels[label_id]
        index = np.where(predict == label_id)

        if sum_method:
            num_of_pixel = len(index[0])
            result["detail"].update({label: num_of_pixel})

            if num_of_pixel >= thr[label_id]:
                predict_plot[index[0], index[1], :] = COLOR_LIST[label_id]
                result['gt_thr'].append(label)

        else:
            mask_index = np.zeros((model_output.shape[1], model_output.shape[2]), dtype=np.uint8)
            mask_index[index[0], index[1]] = 255
            cnts, _ = cv2.findContours(mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw_cns = []
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area >= thr[label_id]:
                    draw_cns.append(cnt)

            cv2.drawContours(predict_plot, draw_cns, -1, tuple(COLOR_LIST[label_id]), -1)

    return result, predict_plot


def fuse_image(
        img_path: str,
        img: np.ndarray,
        mask: np.ndarray,
        save: str
) -> None:
    basename = os.path.basename(img_path)
    name, suffix = os.path.splitext(basename)
    img_fuse = cv2.addWeighted(img, 0.3, mask, 0.5, 0)
    shutil.copy(img_path, save)
    cv2.imwrite(os.path.join(save, name + '_seg.jpg'), img_fuse)


def draw(
        batch_detection: List[Detection],
        images: List[np.ndarray],
        images_path: List[str],
        labels: List[str],
        save: str,
        draw_label: Optional[bool] = True
) -> None:
    if not os.path.exists(save):
        os.makedirs(save)

    image: np.ndarray
    prediction: Detection
    for i, (image, prediction) in enumerate(zip(images, batch_detection)):

        if prediction.is_empty:
            continue

        image_path: str = images_path[i]
        draw_image: np.ndarray = deepcopy(image)

        box: Box
        for box in prediction.boxes:
            color: list = COLOR_LIST[box.label_id].tolist()
            label: str = labels[box.label_id]
            draw_box(
                img=draw_image,
                box=box.location.raw(),
                score=box.score,
                label=label,
                color=color,
                draw_label=draw_label
            )

        basename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(save, basename), draw_image)


def draw_box(
        img: np.ndarray,
        box: List[int],
        score: float,
        label: str,
        color: list,
        draw_label: Optional[bool] = True,
) -> None:
    x1, y1, w, h = box

    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x1 + w), int(y1 + h)),
        color,
        2
    )

    if draw_label:
        label = f"{label}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED
        )
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


def yolov8_postprocess(  # noqa
        outputs: np.ndarray,  # shape:[n,c]
        batch_image_path: List[str],
        input_wh: list,
        image_wh: list,
        iou: Optional[float] = 0.5,
        conf: Optional[float] = 0.5
) -> List[Detection]:
    batch_detection = []
    for bs in range(outputs.shape[0]):
        image_path = batch_image_path[bs]
        output: np.ndarray = outputs[bs]
        output: np.ndarray = np.transpose(output)

        rows: int = output.shape[0]

        boxes = []
        scores = []
        label_ids = []

        x_factor: float = image_wh[0] / input_wh[0]
        y_factor: float = image_wh[1] / input_wh[1]

        detection = Detection()
        for i in range(rows):
            classes_scores: float = output[i][4:]

            max_score = np.amax(classes_scores)

            if max_score >= conf:
                class_id = int(np.argmax(classes_scores))

                x, y, w, h = output[i][0], output[i][1], output[i][2], output[i][3]

                x0 = int((x - w / 2) * x_factor)
                y0 = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                label_ids.append(class_id)
                scores.append(max_score)
                boxes.append([x0, y0, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
        detection.image = image_path
        for i in indices:
            detection.boxes.append(Box(
                location=Location(*boxes[i]),
                score=scores[i],
                label_id=label_ids[i]
            ))
        if len(detection) != 0:
            detection.is_empty = False

        batch_detection.append(detection)

    return batch_detection


def yolov10_postprocess(
        outputs: np.ndarray,
        batch_image_path: List[str],
        input_wh: List[int],
        image_wh: List[int],
        conf: Optional[float] = 0.5
) -> List[Detection]:
    x_factor: float = image_wh[0] / input_wh[0]
    y_factor: float = image_wh[1] / input_wh[0]

    batch_detection: List[Detection] = []
    for bs in range(outputs.shape[0]):
        image_path = batch_image_path[bs]
        output: np.ndarray = outputs[bs]

        rows, cols = output.shape

        detection = Detection()
        detection.image = image_path

        for i in range(rows):
            score = round3(output[i, 4])

            if score >= conf:
                left, top = output[i][0].item(), output[i][1].item()
                right, bottom = output[i][2].item(), output[i][3].item()

                x1 = int(left * x_factor)
                y1 = int(top * y_factor)
                x2 = int(right * x_factor)
                y2 = int(bottom * y_factor)

                xywh = xyxy2xywh([x1, y1, x2, y2])

                box = Box(
                    Location(*xywh),
                    score=score,
                    label_id=int(output[i, 5])
                )
                detection.boxes.append(box)
        batch_detection.append(detection)
    return batch_detection
