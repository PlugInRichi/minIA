import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

BINOMIAL_KERNEL = 1 / 273 * np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]])

def filter_image(image, kernel=BINOMIAL_KERNEL):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.medianBlur(img_gray, 9)
    img_open = cv.morphologyEx(img_blur, op=cv.MORPH_OPEN, kernel=kernel, iterations=3)
    (thresh, im_bw) = cv.threshold(img_open, 8, 255, cv.THRESH_BINARY)
    return cv.bitwise_and(image, image, mask=im_bw)


def change_position(image):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    eighth_height, eighth_width = height / 8, width / 8
    sng = 1 if random.random() < 0.5 else -1
    T = np.float32([
            [1, 0, sng * eighth_width],
            [0, 1, sng * eighth_height]])
    R = cv.getRotationMatrix2D(center, angle=sng*30, scale=1)
    img_translation = cv.warpAffine(image, T, (width, height))
    img_rotation = cv.warpAffine(img_translation, R, (width, height))
    img_flip = cv.flip(img_rotation, sng)
    return img_flip


def filter_images(list_im_names, dir_path, dest_dir):
    """

    Args:
        list_im_names: List of names without file extension
        dir_path: Source path directory
        dest_dir: Destiny path directory
    """
    for im_name in tqdm(list_im_names):
        path = os.path.join(dir_path, im_name + '.jpg')
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("La imagen " + im_name + " no existe")
        im_new_pos = change_position(image)
        im_filtered = filter_image(im_new_pos)
        cv.imwrite(os.path.join(dest_dir, 'F' + im_name + '.jpg'), im_filtered)