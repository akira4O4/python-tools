import os
import shutil
from utils import get_images
from tqdm import tqdm

if __name__ == '__main__':
    """
    if check_dir.image in target_dir.all_images:
       remove check_dir.image
    """

    check_dir = r''
    target_dir = r''
    output = r''
    mode = 'move'

    check_images = get_images(check_dir)
    target_images = get_images(target_dir)

    print(f'images1: {len(check_images)}')
    print(f'images2: {len(target_images)}')

    count = 0
    func = None
    target_images = [os.path.basename(img) for img in target_images]

    for image in tqdm(check_images):
        basename = os.path.basename(image)
        if basename in target_images:
            count += 1

            if mode == 'copy':
                func = shutil.copy
            elif mode == 'move':
                func = shutil.move

            try:
                func(image, output)
            except:
                func(image, os.path.join(output, basename + ".dup"))

    print(f'copy or move {count} images.')
