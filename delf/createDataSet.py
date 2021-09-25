import os
import pandas as pd
import numpy as np
import cv2 as cv
import random

debiased = [
    'dr7objid',
    't08_odd_feature_a19_ring_debiased',
    't08_odd_feature_a20_lens_or_arc_debiased',
    't08_odd_feature_a21_disturbed_debiased',
    't08_odd_feature_a22_irregular_debiased',
    't08_odd_feature_a24_merger_debiased',
    't08_odd_feature_a38_dust_lane_debiased',

    't11_arms_number_a31_1_debiased',
    't11_arms_number_a32_2_debiased',
    't11_arms_number_a33_3_debiased',
    't11_arms_number_a34_4_debiased',
    't05_bulge_prominence_a10_no_bulge_debiased',
    't05_bulge_prominence_a11_just_noticeable_debiased',
    't05_bulge_prominence_a12_obvious_debiased',
    't05_bulge_prominence_a13_dominant_debiased',

    't07_rounded_a16_completely_round_debiased',
    't07_rounded_a17_in_between_debiased',
    't07_rounded_a18_cigar_shaped_debiased',

    't09_bulge_shape_a25_rounded_debiased',
    't09_bulge_shape_a26_boxy_debiased',
    't09_bulge_shape_a27_no_bulge_debiased',

    't10_arms_winding_a28_tight_debiased',
    't10_arms_winding_a29_medium_debiased',
    't10_arms_winding_a30_loose_debiased']


def getBestScores(df, threshold):
    """
    Get the classes with una confiabilidad superior a threshold
    """
    df2 = df[debiased].copy()
    df2 = df2[df2 > threshold]
    df2.columns = range(len(df2.columns))
    return df2


def imagesPerClass(df):
    """
    Create a list of lists where every sublist is the names of the images assigned to that class
    """
    classes = list()
    for class_ in range(1, len(df.columns)):
        images_class = df[[0, class_]].dropna()[0].astype(str).tolist()  # Hacer un downsapling para grnades
        classes.append(images_class)
    return classes


def downsamplig(classes):
    random.seed(505)
    random.shuffle(classes[14])
    random.shuffle(classes[15])
    random.shuffle(classes[17])
    classes[14] = classes[14][0:len(classes[14])//2]
    classes[15] = classes[15][0:len(classes[15])//3]
    classes[17] = classes[17][0:(len(classes[14])//5)*3]

def createTrainingFile(ruta, images_class):
    id_classes = range(0, len(images_class))
    with open(ruta, 'w') as dataset:
        dataset.write('type_galaxy_id,images\n')
        for class_id, img_class in zip(id_classes, images_class):
            images = ' '.join(img_class)
            dataset.write(str(class_id) + ',' + images+'\n')


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
    quarter_height, quarter_width = height / 8, width / 8
    T = np.float32([
            [1, 0, quarter_width],
            [0, 1, quarter_height]])
    R = cv.getRotationMatrix2D(center, angle=30, scale=1)
    img_translation = cv.warpAffine(image, T, (width, height))
    img_rotation = cv.warpAffine(image, R, (width, height))
    img_flip = cv.flip(img_rotation, -1)
    return img_flip


def createImageDataSet(images, dir_path):
    kernel = np.ones((3, 3), np.uint8)
    for image_name in images:
        path = os.path.join(dir_path, image_name + '.jpg')
        image = cv.imread(path, cv.IMREAD_COLOR)
        img_filtered = filterImg(image, kernel, 6)
        new_image = changePosition(img_filtered)
        cv.imwrite(os.path.join(dir_path, image_name + 'F' + '.jpg'), new_image)


if __name__ == '__main__':
    galaxyZoo2 = r'/home/rick/Proyectos/minIA.old/notebooks/DataCharacterization/zoo2MainSpecz.csv'
    train_file = r'/home/rick/Proyectos/minIA/delf/GZ2_classes.csv'
    images_dir = r''

    df_g = pd.read_csv(galaxyZoo2)
    df_img = getBestScores(df_g, 0.8)  # use df for testing
    img_per_class = imagesPerClass(df_img)
    downsamplig(img_per_class)
    images = set()
    for img_class in img_per_class:
        images += set(img_class)
    createImageDataSet(images, images_dir)
    createTrainingFile(train_file, img_per_class)

