from os import listdir, path
from collections import Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv

plt.rcParams['axes.facecolor'] = 'black'
def load_files(dir_path):
    files = listdir(dir_path)
    return dict([(file, path.join(dir_path, file)) for file in files])


def load_model(model_path):
    global structures_size
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
        im_features_df.set_index('image_name', drop=False, inplace=True)
        im_features_df.index.rename('image_id', inplace=True)
    return im_features_df

def search_structure(im_features_df, structure_id, threshold, split=0.01):
    max_features = int(len(im_features_df) * split)
    cnt_vw_struct = Counter(dict(structures[structure_id]))
    num_features_struct = sum(cnt_vw_struct.values())
    match_imgs = im_features_df.iloc[0:max_features]
    match_imgs = match_imgs['visual_word_id'].isin(cnt_vw_struct.keys())
    match_imgs = set(match_imgs[match_imgs].index)
    images_df = im_features_df.loc[match_imgs].copy()
    images_df['overlap'] = 0
    vws_per_image = images_df[['image_name', 'visual_word_id']].groupby('image_name').apply(pd.DataFrame.to_records,
                                                                                           index=False)
    print('Searching in ', len(vws_per_image), 'images...')
    for vws_image in tqdm(vws_per_image):
        img_name, img_vw_ids = zip(*vws_image)
        cnt_vw_im = Counter(img_vw_ids)
        intersection = sum((cnt_vw_struct & cnt_vw_im).values())
        num_features_im = len(img_vw_ids)
        overlap = intersection / min(num_features_im, num_features_struct)
        if overlap < threshold:
            images_df.drop(labels=img_name[0], inplace=True)
        else:
            images_df.loc[img_name[0], 'overlap'] = overlap
    print(len(images_df.groupby('image_name').count()), ' images found.')
    return images_df


# Enviar máximo 9 imágenes
# img_names is a dictionary name/ path
def show_images_per_structure(images_df, structure_id, img_names, im_index, path_images, gray=False):
    start = im_index - 1
    end = im_index + 8
    img_selected = img_names[start:end] if end < len(img_names) else img_names[start:]
    num_images = len(img_selected)   
    if num_images <= 3:
        x_fig, y_fig = 1, num_images
    elif num_images <= 6:
        x_fig, y_fig = 2, 3
    elif num_images <= 9:
        x_fig, y_fig = 3, 3
    else:
        raise ValueError("Number of image names cannot be greater than 9")
    fig, axs = plt.subplots(x_fig, y_fig, figsize=(15, 15), dpi=200)
    vw_struct = dict(structures[structure_id]).keys()
    i = 0
    for img_name in img_selected: #Limitar las imágenes con el valor de i
        img_match_df = images_df.loc[img_name]
        try:
            img_match_df = img_match_df[img_match_df['visual_word_id'].isin(vw_struct)]
            points = zip(img_match_df['location_x'],
                         img_match_df['location_y'],
                         img_match_df['size'])
            overlap = img_match_df['overlap'].iloc[0]
        except AttributeError:
            points = [(img_match_df['location_x'],
                       img_match_df['location_y'],
                       img_match_df['size'])]
            overlap = img_match_df["overlap"]

        color_schema = cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB
        img_name = str(img_name)
        img_name = img_name[1:] if img_name[0] == 'F' else img_name
        img = cv.imread(path.join(path_images, img_name+'.jpg'))
        img = cv.cvtColor(img, color_schema)

        for x, y, size in points:
            img = cv.circle(img, (int(x), int(y)), int(size//2), color=(255, 0, 255), thickness=1, lineType=0, shift=0)

        y_pos = i % 3
        x_pos = i // 3
        if y_fig > 1:
            axs[x_pos, y_pos].set_xticklabels([])
            axs[x_pos, y_pos].xaxis.label.set_color('white')
            axs[x_pos, y_pos].set_yticklabels([])
            axs[x_pos, y_pos].set_xlabel(f'{img_name}\noverlap: {overlap:0.3}')
            axs[x_pos, y_pos].imshow(img)
        else:
            axs[x_pos].set_xticklabels([])
            axs[x_pos].xaxis.label.set_color('white')
            axs[x_pos].set_yticklabels([])
            axs[x_pos].set_xlabel(f'{img_name}\noverlap: {overlap:0.3}')
            axs[x_pos].imshow(img)
        i += 1
        #plt.savefig('/data/images/DELF_extractor/prueba',facecolor=(2/255,2/255,2/255),bbox_inches='tight')

if __name__ == '__main__':
    models = load_files('/data/SMH_models')
    features = load_files('/data/visual_vocabulary')
    load_model(models['simply_features80_DELF_vv.model'])
    load_im_features(features['simply_features80_DELF_vv.csv'])
    images_df = search_structure(0, 0.5, 0.05)
    img_names = images_df['image_name'].unique()
    show_images_per_structure(images_df, 0, img_names[:-2], gray=False)
    pass
