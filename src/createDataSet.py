import os
import pandas as pd
import numpy as np
import cv2 as cv
import yaml
from tqdm import tqdm
from minIA.imageProcessing import filter_image, change_position
from minIA.constrains import SELECTED_TYPES, PRIOR_TYPES, GALAXY_TYPES, ALL_TYPES

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
    #found_diff = list(images_found - set(df_clean['asset_id']))
    #for image_lost in found_diff:
    #    os.remove(images_path + image_lost + '.jpg')
    indexes = df_clean['asset_id'].isin(map_diff)
    return df_clean.loc[indexes]

def assign_class(df, threshold=0.80):
    """return SET OF DATAFRAMES"""
    galaxy_types = {}
    #Pink
    odd_yes = df['t06_odd_a14_yes_debiased'] > 0.9
    galaxy_types['ring'] = df['asset_id'][odd_yes & df['t08_odd_feature_a19_ring_debiased'] > threshold]
    galaxy_types['lens_or_arc'] = df['asset_id'][odd_yes & df['t08_odd_feature_a20_lens_or_arc_debiased'] > threshold]
    galaxy_types['disturbed'] = df['asset_id'][odd_yes & df['t08_odd_feature_a21_disturbed_debiased'] > threshold]
    galaxy_types['irregular'] = df['asset_id'][odd_yes & df['t08_odd_feature_a22_irregular_debiased'] > threshold]
    galaxy_types['merger'] = df['asset_id'][odd_yes & df['t08_odd_feature_a24_merger_debiased'] > threshold]
    galaxy_types['dust_lane'] = df['asset_id'][odd_yes & df['t08_odd_feature_a38_dust_lane_debiased'] > threshold]

    #Red
    odd_no =df['t06_odd_a15_no_debiased']> 0.9
    galaxy_types['rounded'] = df['asset_id'][odd_no & df['t09_bulge_shape_a25_rounded_debiased'] > 0.90]
    galaxy_types['boxy'] = df['asset_id'][odd_no & df['t09_bulge_shape_a26_boxy_debiased'] > 0.90]
    galaxy_types['no_bulge'] = df['asset_id'][odd_no & df['t09_bulge_shape_a27_no_bulge_debiased'] > 0.90]

    #Orange
    no_spiral = odd_no & df['t04_spiral_a09_no_spiral_debiased'] > 0.9
    galaxy_types['no_central_bulge'] = df['asset_id'][no_spiral &
                                                      df['t05_bulge_prominence_a10_no_bulge_debiased'] > threshold]
    galaxy_types['dominant'] = df['asset_id'][no_spiral & df['t05_bulge_prominence_a13_dominant_debiased'] > threshold]

    #Green
    spiral_yes = odd_no & df['t04_spiral_a08_spiral_debiased'] > 0.9
    galaxy_types['tight'] = df['asset_id'][spiral_yes & df['t10_arms_winding_a28_tight_debiased'] > threshold]
    galaxy_types['medium'] = df['asset_id'][spiral_yes & df['t10_arms_winding_a29_medium_debiased'] > threshold]
    galaxy_types['loose'] = df['asset_id'][spiral_yes & df['t10_arms_winding_a30_loose_debiased'] > threshold]

    #Yellow
    galaxy_types['completely_round'] = df['asset_id'][odd_no &
                                                      df['t07_rounded_a16_completely_round_debiased'] > 0.95]
    galaxy_types['in_between'] = df['asset_id'][odd_no & df['t07_rounded_a17_in_between_debiased'] > 0.95]
    galaxy_types['cigar_shaped'] = df['asset_id'][odd_no & df['t07_rounded_a18_cigar_shaped_debiased'] > 0.95]
    return galaxy_types

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


def down_sampling(galaxy_types):
    used_images = set()
    estimated_images = 0 #Use for find overlapping
    for galaxy_type in GALAXY_TYPES:
        if len(galaxy_types[galaxy_type]) > 2000:
            galaxy_types[galaxy_type] = list(np.random.choice(galaxy_types[galaxy_type], 2000, replace=False))
        else:
            galaxy_types[galaxy_type] = galaxy_types[galaxy_type].to_list()
        used_images |= set(galaxy_types[galaxy_type])
        estimated_images += len(galaxy_types[galaxy_type])
        print('Found ' + str(len(galaxy_types[galaxy_type])) + ' images for ' + galaxy_type + ' class')
    print('Total of images used for training: ', len(used_images),' estimated: ', estimated_images)
    return used_images



def create_training_file(full_path, filtered_path, galaxy_types):
    g_index = 0
    with open(full_path, 'w') as full_dataset, open(filtered_path, 'w') as filtered_dataset:
        full_dataset.write('type_galaxy_id,images\n')
        filtered_dataset.write('type_galaxy_id,images\n')
        for galaxy_type in GALAXY_TYPES:
            print(galaxy_type)
            names = 'F' + ' F'.join(galaxy_types[galaxy_type])
            filtered_dataset.write(str(g_index) + ',' + names + '\n')
            names = names + ' ' + ' '.join(galaxy_types[galaxy_type])
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
    #mask = create_mask(df_clean, cfg_dataset['th_score'])
    #df_clean = df_clean[mask]

    print('Total of samples: ', len(df_clean))
    #if 'sample_size' in cfg_dataset:
    #    print('Using a sample of size ', cfg_dataset['sample_size'])
    #    df_clean = df_clean.sample(cfg_dataset['sample_size'], random_state=6564)

    print('Making some groups... ')
    #galaxy_groups = images_per_class(df_clean, cfg_dataset['th_score'])
    galaxy_groups = assign_class(df_clean)
    used_images = down_sampling(galaxy_groups)

    print('Creando imágenes filtradas... ')
    images = list(used_images)
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
