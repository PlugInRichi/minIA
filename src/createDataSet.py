import os
import pandas as pd
import numpy as np
import cv2 as cv
import yaml
from tqdm import tqdm
from minIA.imageProcessing import filter_image, change_position
from minIA.constrains import SELECTED_TYPES, CLASS_NAMES


def create_dataframe(cfg_dataset):
    df_map = pd.read_csv(cfg_dataset['map_images_path'])
    df_selected = pd.read_csv(cfg_dataset['galaxyZoo2_path'])[SELECTED_TYPES]
    df_selected = df_selected.rename(columns={'dr7objid': 'objid'})
    df_match = df_map.merge(df_selected, on='objid', how='inner')
    df_match['asset_id'] = df_match['asset_id'].astype('string')
    return df_match


def delete_not_found(images_path, df_clean):
    images_found = set([image_name.split('.')[0] for image_name in os.scandir(images_path)])
    map_diff = set(df_clean['asset_id']) - images_found
    found_diff = list(images_found - set(df_clean['asset_id']))
    for image_lost in found_diff:
        os.remove(images_path + image_lost + '.jpg')
    indexes = df_clean['asset_id'].isin(map_diff)
    return df_clean.loc[indexes]


def images_per_class(df_clean):
    """
    Create a list of lists where every sublist is the names of the images assigned to that class
    """
    classes = {}
    for galaxy_type in SELECTED_TYPES:
        classes[galaxy_type] = df_clean['asset_id'][df_clean[galaxy_type] > 0.8]
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
            new_image = change_position(img_filtered)
            cv.imwrite(os.path.join(dir_path, 'F' + image_name + '.jpg'), new_image)
            in_image_dir.add(image_name)
    print(len(in_image_dir), '/', len(all_images))
    return in_image_dir


def main():
    with open("../data/config/dataset_config.yml", "r") as config_file:
        cfg_dataset = yaml.safe_load(config_file)

    print('Creating Dataframe... ')
    df_clean = create_dataframe(cfg_dataset)

    print('Deleting lost images... ')
    df_clean = delete_not_found(cfg_dataset['images_dir_path'], df_clean)

    print('Making some groups... ')
    galaxy_groups = images_per_class(df_clean)
    downsampling(galaxy_groups)

    print('Creando imágenes filtradas... ')
    images = [set(img_class) for img_class in galaxy_groups]
    in_image_dir = create_image_dataset(images, cfg_dataset['images_dir_path'])

    print('Exportando archivo de entrenamiento... ')
    create_training_file(cfg_dataset['train_dataset_path'], images, in_image_dir)
    print('Hecho!')


if __name__ == '__main__':
    import os

    os.chdir('/')

    main()
