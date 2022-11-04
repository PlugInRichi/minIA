import pandas as pd
import numpy as np
import yaml
from minIA.imageProcessing import upsampling_imgs
from minIA.constrains import GALAXY_TYPES, ALL_TYPES
import os

np.random.seed(5986)


def create_dataframe(cfg_dataset):
    df_map = pd.read_csv(cfg_dataset['map_images_path'])
    df_selected = pd.read_csv(cfg_dataset['galaxyZoo2_path']).rename(columns={'dr7objid': 'objid'})
    df_selected = df_selected[ALL_TYPES]
    df_match = df_map.merge(df_selected, on='objid', how='inner')
    df_match['asset_id'] = df_match['asset_id'].astype('string')
    return df_match


def delete_not_found(images_path, df_clean):
    images_found = set([image_name.split('.')[0] for image_name in os.listdir(images_path)])
    map_diff = set(df_clean['asset_id']) & images_found
    indexes = df_clean['asset_id'].isin(map_diff)
    return df_clean.loc[indexes]


def assign_class(df):
    """return SET OF DATAFRAMES"""
    tr_prev = 0.70
    tr_class = 0.75
    galaxy_types = {}

    smooth = df['t01_smooth_or_features_a01_smooth_debiased'] > tr_prev
    features_or_disk = df['t01_smooth_or_features_a02_features_or_disk_debiased'] > tr_prev

    edgeon_yes = features_or_disk & (df['t02_edgeon_a04_yes_debiased'] > tr_prev)
    edgeon_no = features_or_disk & (df['t02_edgeon_a05_no_debiased'] > tr_prev)

    spiral_yes = edgeon_no & (df['t04_spiral_a08_spiral_debiased'] > tr_prev)

    odd_yes = df['t06_odd_a14_yes_debiased'] > tr_prev

    # Pink
    galaxy_types['ring'] = df['asset_id'][odd_yes & (df['t08_odd_feature_a19_ring_debiased'] > tr_class)]

    # Red
    galaxy_types['rounded'] = df['asset_id'][edgeon_yes & (df['t09_bulge_shape_a25_rounded_debiased'] > tr_class)]
    galaxy_types['no_bulge'] = df['asset_id'][edgeon_yes & (df['t09_bulge_shape_a27_no_bulge_debiased'] > tr_class)]

    # Green
    galaxy_types['tight'] = df['asset_id'][spiral_yes & (df['t10_arms_winding_a28_tight_debiased'] > tr_class)]
    galaxy_types['medium'] = df['asset_id'][spiral_yes & (df['t10_arms_winding_a29_medium_debiased'] > tr_class)]
    galaxy_types['loose'] = df['asset_id'][spiral_yes & (df['t10_arms_winding_a30_loose_debiased'] > tr_class)]

    # Yellow
    galaxy_types['completely_round'] = df['asset_id'][smooth &
                                                      (df['t07_rounded_a16_completely_round_debiased'] > tr_class)]
    galaxy_types['in_between'] = df['asset_id'][smooth & (df['t07_rounded_a17_in_between_debiased'] > tr_class)]
    galaxy_types['cigar_shaped'] = df['asset_id'][smooth & (df['t07_rounded_a18_cigar_shaped_debiased'] > tr_class)]
    return galaxy_types


# Just for testing
def overlapping(dic_galaxy):
    dic_galaxy_c = dic_galaxy.copy()
    for key, value in dic_galaxy.items():
        for key_2, value_2 in dic_galaxy_c.items():
            op = len(set(value) & set(value_2))
            if op != 0:
                print('Overlapping between ', key, ' and ', key_2, ' is ', op)
        dic_galaxy_c.pop(key)


def images_per_class(df_clean, threshold):
    """
    Create a list of lists where every sublist is the names of the images assigned to that class
    """
    classes = {}
    for galaxy_type in GALAXY_TYPES:
        classes[galaxy_type] = df_clean['asset_id'][df_clean[galaxy_type] > threshold].to_list()
    return classes


def down_sampling(galaxy_types, class_size):
    used_images = set()
    estimated_images = 0  # Use for find overlapping
    for galaxy_type in GALAXY_TYPES:
        if galaxy_type in galaxy_types.keys():
            if len(galaxy_types[galaxy_type]) > class_size:
                galaxy_types[galaxy_type] = list(np.random.choice(galaxy_types[galaxy_type], class_size, replace=False))
            else:
                galaxy_types[galaxy_type] = galaxy_types[galaxy_type].to_list()
            used_images |= set(galaxy_types[galaxy_type])
            estimated_images += len(galaxy_types[galaxy_type])
            print('Found ' + str(len(galaxy_types[galaxy_type])) + ' images for ' + galaxy_type + ' class')
    print('Total of images used for training: ', len(used_images), ' estimated: ', estimated_images)
    return used_images


def create_training_file(train_dataset_path, galaxy_types):
    g_index = 0
    original_dataset_path = train_dataset_path+'_original.csv'
    merged_dataset_path = train_dataset_path+'_with_filter.csv'
    with open(original_dataset_path, 'w') as full_dataset, open(merged_dataset_path, 'w') as filtered_dataset:
        full_dataset.write('type_galaxy_id,images\n')
        filtered_dataset.write('type_galaxy_id,images\n')
        for galaxy_type in GALAXY_TYPES:
            if galaxy_type in galaxy_types.keys():
                print(galaxy_type)
                names_filtered = 'F' + ' F'.join(galaxy_types[galaxy_type])
                names_augmentation = 'A' + ' A'.join(galaxy_types[galaxy_type])

                names = names_filtered + ' ' + ' '.join(galaxy_types[galaxy_type])
                filtered_dataset.write(str(g_index) + ',' + names + '\n')

                names = names_augmentation + ' ' + ' '.join(galaxy_types[galaxy_type])
                full_dataset.write(str(g_index) + ',' + names + '\n')
                g_index += 1


def main():
    with open("../data/config/dataset_config.yml", "r") as config_file:
        cfg_dataset = yaml.safe_load(config_file)

    print('Creating Dataframe... ')
    df_clean = create_dataframe(cfg_dataset)

    print('Deleting lost images... ')
    df_clean = delete_not_found(cfg_dataset['images_dir_path'], df_clean)
    print('Total of samples: ', len(df_clean))

    print('Making some groups... ')
    galaxy_groups = assign_class(df_clean)
    used_images = down_sampling(galaxy_groups, cfg_dataset['class_size'])

    print('Creando imágenes filtradas... ')
    upsampling_imgs(list(used_images), cfg_dataset['images_dir_path'], cfg_dataset['images_out_dir_path'])
    upsampling_imgs(list(used_images), cfg_dataset['images_dir_path'], cfg_dataset['images_out_dir_path'], False)

    print('Exportando archivo de entrenamiento... ')
    create_training_file(cfg_dataset['train_dataset_path'], galaxy_groups)
    print('Hecho!')


if __name__ == '__main__':
    os.chdir('/')
    main()
