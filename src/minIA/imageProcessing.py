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

MORPH_ELLIPSE_5x5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
MORPH_ELLIPSE_11x11 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))


def filter_image(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.medianBlur(img_gray, 3)
    (thresh, im_bw) = cv.threshold(img_blur, 11, 255, cv.THRESH_BINARY)
    img_erode = cv.erode(im_bw, kernel=MORPH_ELLIPSE_5x5, iterations=3)
    img_dilate = cv.dilate(img_erode, kernel=MORPH_ELLIPSE_11x11, iterations=2)
    return cv.bitwise_and(image, image, mask=img_dilate)


def change_position(image):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    eighth_height, eighth_width = height / 8, width / 8
    sng = 1 if random.random() < 0.5 else -1
    T = np.float32([
        [1, 0, sng * eighth_width],
        [0, 1, sng * eighth_height]])
    R = cv.getRotationMatrix2D(center, angle=sng * 30, scale=1)
    img_translation = cv.warpAffine(image, T, (width, height))
    img_rotation = cv.warpAffine(img_translation, R, (width, height))
    img_flip = cv.flip(img_rotation, sng)
    return img_flip


def upsampling_imgs(list_im_names, dir_path, dest_dir, filter=True):
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
        new_im, prefix = (filter_image(im_new_pos), 'F') if filter else (im_new_pos, 'A')
        cv.imwrite(os.path.join(dest_dir, prefix + im_name + '.jpg'), new_im)
