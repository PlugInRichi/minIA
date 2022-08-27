import os
import pandas as pd
import numpy as np
import cv2 as cv
import yaml
from tqdm import tqdm
from minIA.imageProcessing import filter_image, change_position
from minIA.constrains import SELECTED_TYPES, PRIOR_TYPES, GALAXY_TYPES


def create_dataframe(cfg_dataset):
    df_map = pd.read_csv(cfg_dataset['map_images_path'])
    df_selected = pd.read_csv(cfg_dataset['galaxyZoo2_path']).rename(columns={'dr7objid': 'objid'})
    df_selected = df_selected[SELECTED_TYPES]
    df_match = df_map.merge(df_selected, on='objid', how='inner')
    df_match['asset_id'] = df_match['asset_id'].astype('string')
    return df_match


def delete_not_found(images_path, df_clean):
    images_found = set([image_name.split('.')[0] for image_name in os.listdir(images_path)])
    map_diff = set(df_clean['asset_id']) & images_found
    #found_diff = list(images_found - set(df_clean['asset_id']))
    #for image_lost in found_diff:
    #    os.remove(images_path + image_lost + '.jpg')
    indexes = df_clean['asset_id'].isin(map_diff)
    return df_clean.loc[indexes]


def create_mask(dataframe, threshold):
    mask = None
    for prior in PRIOR_TYPES:
        if mask is None:
            mask = dataframe[prior] > threshold
        else:
            mask |= dataframe[prior] > threshold
    return mask

def images_per_class(df_clean, threshold):
    """
    Create a list of lists where every sublist is the names of the images assigned to that class
    """
    classes = {}
    for galaxy_type in GALAXY_TYPES:
        classes[galaxy_type] = df_clean['asset_id'][df_clean[galaxy_type] > threshold].to_list()
    return classes


def downsampling(classes):
    for i in range(len(classes)):
        if len(classes[i]) > 10000:
            classes[i] = np.random.choice(classes[i], 10000, replace=False)


def create_training_file(full_path, filtered_path, images_class):
    g_index = 0
    with open(full_path, 'w') as full_dataset, open(filtered_path, 'w') as filtered_dataset:
        full_dataset.write('type_galaxy_id,images\n')
        filtered_dataset.write('type_galaxy_id,images\n')
        for galaxy_type in GALAXY_TYPES:
            print(galaxy_type)
            names = 'F' + ' F'.join(images_class[galaxy_type])
            filtered_dataset.write(str(g_index) + ',' + names + '\n')
            names = names + ' ' + ' '.join(images_class[galaxy_type])
            full_dataset.write(str(g_index) + ',' + names + '\n')
            g_index += 1


def create_image_dataset(images, dir_path):
    kernel = np.ones((3, 3), np.uint8)
    in_image_dir = set()
    all_images = images
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

    print('Applying mask to prioritize underrepresented classes...')
    mask = create_mask(df_clean, cfg_dataset['th_score'])
    df_clean = df_clean[mask]
    print('Total of calculated samples: ', len(df_clean))
    if 'sample_size' in cfg_dataset:
        print('Using a sample of size ', cfg_dataset['sample_size'])
        df_clean = df_clean.sample(cfg_dataset['sample_size'], random_state=6564)

    print('Making some groups... ')
    galaxy_groups = images_per_class(df_clean, cfg_dataset['th_score'])

    print('Creando imágenes filtradas... ')
    images = df_clean['asset_id'].to_list()
    in_image_dir = create_image_dataset(images, cfg_dataset['images_dir_path']) #faltaron 3?

    print('Exportando archivo de entrenamiento... ')
    create_training_file(cfg_dataset['full_train_dataset_path'],
                         cfg_dataset['filtered_train_dataset_path'],
                         galaxy_groups)
    print('Hecho!')


if __name__ == '__main__':
    import os

    os.chdir('/')

    main()
