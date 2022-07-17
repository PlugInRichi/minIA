import random
import cv2 as cv
import numpy as np


def filterImg(image, kernel, th=6):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = cv.erode(img_gray, kernel, iterations=3)
    mask = cv.blur(mask, (5, 5))
    mask[mask <= th] = 0
    mask[mask > 0] = 1
    return cv.bitwise_and(image, image, mask=mask)


def changePosition(image):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    eighth_height, eighth_width = height / 8, width / 8
    sng = 1 if random.random() < 0.5 else -1
    T = np.float32([
            [1, 0, sng * eighth_width],
            [0, 1, sng * eighth_height]])
    R = cv.getRotationMatrix2D(center, angle=sng*30, scale=1)
    img_translation = cv.warpAffine(image, T, (width, height))
    img_rotation = cv.warpAffine(image, R, (width, height))
    img_flip = cv.flip(img_rotation, sng)
    return img_flip
