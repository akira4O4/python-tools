import os
import cv2
from tqdm import tqdm
from utils import get_images

root = r'r'
output = r''
if not os.path.exists(output):
    os.makedirs(output)

wh = [1280, 1024]
for image in tqdm(get_images(root)):
    basename = os.path.basename(image)
    im = cv2.imread(image)
    if im is None:
        print(f'{image} can`t open.')
        continue
    im_resize = cv2.resize(im, wh)
    cv2.imwrite(os.path.join(output, basename), im_resize)
