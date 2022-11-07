from os import listdir, path
from collections import Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv


def load_files(dir_path):
    files = listdir(dir_path)
    return dict([(file, path.join(dir_path, file)) for file in files])


def load_model(model_path):
    global models_size
    global structures
    with open(model_path, 'r') as model_file:
        model = model_file.readlines()
    structures_size = []
    structures = []
    for line in model:
        structure = line.strip().split()
        structures_size.append(int(structure[0]))
        structure = [tuple(map(int, visual_word.split(':'))) for visual_word in structure[1:]]
        structures.append(structure)
    print('#Structures: ' + str(len(structures)))
    print('Model name:', path.split(model_path)[-1])
    return structures_size, structures


def load_im_features(im_features_path):
    global im_features_df
    with open(im_features_path, 'r') as file:
        im_features_df = pd.read_csv(file)
        im_features_df.drop('index', axis=1, inplace=True)


def search_structure(structure_id, threshold, split=0.01):
    global im_features_df
    max_features = int(len(im_features_df) * split)
    cnt_vw_struct = Counter(dict(structures[structure_id]))
    num_features_struct = sum(cnt_vw_struct.values())
    images_df = im_features_df.iloc[:max_features].copy()
    vws_per_image = images_df[['image_name', 'descriptor_id']].groupby('image_name').apply(pd.DataFrame.to_records,
                                                                                           index=False)
    images_df.set_index('image_name', inplace=True, drop=False)
    print('Searching in ', len(vws_per_image), 'images...')
    for vws_image in tqdm(vws_per_image):
        img_name, vws_id = zip(*vws_image)
        cnt_vw_im = Counter(vws_id)
        intersection = sum((cnt_vw_struct & cnt_vw_im).values())
        num_features_im = len(vws_id)
        overlap = intersection / min(num_features_im, num_features_struct)
        if overlap < threshold:
            images_df.drop(labels=img_name[0], inplace=True)
    return images_df


# Enviar máximo 9 imágenes
# img_names is a dictionary name/ path
def show_images_per_structure(images_df, structure_id, img_names, gray=True):
    num_images = len(img_names)
    if num_images <= 3:
        x, y = 1, num_images
    elif num_images <= 6:
        x, y = 2, 3
    elif num_images <= 9:
        x, y = 3, 3
    else:
        raise ValueError("Number of image names cannot be greater than 9")
    fig, axs = plt.subplots(x, y)

    cnt_vw_struct = Counter(dict(structures[structure_id]))
    num_features_struct = sum(cnt_vw_struct.values())

    i = 0
    for img_name in img_names.keys():
        img_features = images_df[images_df['image_name'] == img_name]
        vws_id = img_features['descriptor_id'].tolist()
        cnt_vw_im = Counter(vws_id)
        intersection = sum((cnt_vw_struct & cnt_vw_im).values())
        num_features_im = len(vws_id)
        overlap = intersection / min(num_features_im, num_features_struct)

        color_schema = cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB
        img = cv.imread(img_names[img_name])
        img = cv.cvtColor(img, color_schema)

        img_features = img_features[img_features['descriptor_id'].isin(cnt_vw_struct.keys())]

        #location_str = images_df['location'].tolist()
        #location_str = [loc.strip("['']").split() for loc in location_str]

        for x, y, size in img_features['location_x'], img_features['location_y'], img_features['size']:  # Multiplicador delf ZIP funciona
            img = cv.circle(img, (x, y), size, color=(255, 0, 0), thickness=1, lineType=0, shift=0)

        x_pos = i % 3
        y_pos = i // 3
        axs[x_pos, y_pos].set_xticklabels([])
        axs[x_pos, y_pos].set_yticklabels([])
        axs[x_pos, y_pos].set_xlabel(f'{i}, {overlap:0.3}')
        axs[x_pos, y_pos].imshow(img)
        i += 1


if __name__ == '__main__':
    models = load_files('/data/SMH_models')
    features = load_files('/data/visual_vocabulary')
    load_model(models['reduced_full_DELF_1000_r2_l1000.model'])
    load_im_features(features['all_features_DELF_1000.csv'])
    images_df = search_structure(10, 0.5, 0.05)
    img_names = images_df['image_name'].unique()
    show_images_per_structure(images_df, 10, img_names[:-2], gray=True)
    pass
