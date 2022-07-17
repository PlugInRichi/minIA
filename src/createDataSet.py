import os
import pandas as pd
import numpy as np
import cv2 as cv
import yaml
from tqdm import tqdm
from minIA.imageProcessing import filter_image, changePosition
from minIA.constrains import DEBIASED, CLASS_NAMES


def get_best_scores(df, threshold):
    """
    Get the classes with una confiabilidad superior a threshold
    """
    df2 = df[DEBIASED].copy()
    df2 = df2[df2 > threshold]
    df2.columns = range(len(df2.columns))
    return df2


def images_per_class(df):
    """
    Create a list of lists where every sublist is the names of the images assigned to that class
    """
    classes = list()
    for class_ in range(1, len(df.columns)):
        images_class = df[[0, class_]].dropna()[0].astype(str).tolist()  # Hacer un downsapling para grnades
        classes.append(images_class)
    return classes


def downsampling(classes):
    for i in range(len(classes)):
        if len(classes[i]) > 10000:
            classes[i] = np.random.choice(classes[i], 10000, replace=False)


def create_training_file(ruta, images_class, in_image_dir):
    id_classes = range(0, len(images_class))
    with open(ruta, 'w') as dataset:
        dataset.write('type_galaxy_id,images\n')
        for class_id, img_class in zip(id_classes, images_class):
            img_class &= in_image_dir
            images = ' '.join(img_class) + ' F' + ' F'.join(img_class)
            dataset.write(str(class_id) + ',' + images + '\n')
            print(len(img_class), ' imágenes encontradas para la clase ', CLASS_NAMES[class_id])


def create_image_dataset(images, dir_path):
    kernel = np.ones((3, 3), np.uint8)
    in_image_dir = set()
    all_images = set()
    for img_class in images:
        all_images |= img_class
    for image_name in tqdm(all_images):
        path = os.path.join(dir_path, image_name + '.jpg')
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is not None:
            img_filtered = filter_image(image, kernel, 6)
            new_image = changePosition(img_filtered)
            cv.imwrite(os.path.join(dir_path, 'F' + image_name + '.jpg'), new_image)
            in_image_dir.add(image_name)
    print(len(in_image_dir), '/', len(all_images))
    return in_image_dir


def main():
    with open("../data/config/dataset_config.yml", "r") as config_file:
        cfg_dataset = yaml.safe_load(config_file)

    print('Obteniendo clases por imagen... ')
    df_data = pd.read_csv(cfg_dataset['galaxyZoo2_path'])
    df_map = pd.read_csv(cfg_dataset['map_images_path'])
    df_data = df_data.join(df_map, lsuffix='_caller', rsuffix='_other')

    df_img = get_best_scores(df_data, cfg_dataset['th_score'])
    img_per_class = images_per_class(df_img)
    downsampling(img_per_class)

    print('Creando imágenes filtradas... ')
    images = [set(img_class) for img_class in img_per_class]
    in_image_dir = create_image_dataset(images, cfg_dataset['images_dir_path'])

    print('Exportando archivo de entrenamiento... ')
    create_training_file(cfg_dataset['train_dataset_path'], images, in_image_dir)
    print('Hecho!')


if __name__ == '__main__':
    main()
