#!/usr/bin/env python
# coding: utf-8

'''
Obtiene el documento necesario para SMH, asiciando el ID de cada
imagen con los ID's de los descriotores (centroides) de la imagen.

Entrada:
    Archivo de descriptores originales (generado por el extractor)
    Archivo de descriptores clusterizados

Salida:
    Documento de entrada para SHM
'''

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cluster",
                    help='Ruta del archivo de descriptores generado por el cluster')
parser.add_argument("document_name",
                    help='Ruta y nombre del documento a generar')
parser.add_argument('-drop_outliers',
                    help='Elimina los descriptores anormales', default=False, action="store_true")
parser.add_argument("-use_size",
                    help='Realiza el conteo utilizando el tamaño del descriptor',
                    default='count', action="store_const", const='sum')
args = parser.parse_args()


def gen_document(descriptors_df, agg_function):
    with open(args.document_name, 'w') as file:
        for name, group in tqdm(descriptors_df.groupby(['image_name'])):
            desc_weight_df = group[['visual_word_id', 'size']].groupby('visual_word_id', as_index=False).agg(agg_function)
            desc_weight = zip(desc_weight_df.iloc[:, 0], desc_weight_df.iloc[:, 1])
            image_desc = [str(desc_id)+':'+str(round(weight)) for desc_id, weight in desc_weight]
            file.write(str(len(image_desc))+' '+' '.join(image_desc)+'\n')


def delete_stop_words(desc_distribution):
    plot_distribution(desc_distribution, 'original_'+str(len(desc_distribution)))
    mean_desc = desc_distribution.mean()
    std_desc = desc_distribution.std()
    low_limit = mean_desc - 2 * std_desc
    upper_limit = mean_desc + 2 * std_desc
    desc_distribution = desc_distribution[desc_distribution > low_limit]
    desc_distribution = desc_distribution[desc_distribution < upper_limit]
    plot_distribution(desc_distribution, '2sigma_'+str(len(desc_distribution)))
    print('\nVisual word mean: ', "{:.2f}".format(mean_desc))
    print('Visual word standard deviation: ', "{:.2f}".format(std_desc))
    return desc_distribution


def reduce_vocabulary(features_df):
    vw_distribution = features_df['visual_word_id'].value_counts()
    vocabulary_size = len(vw_distribution)
    vw_total = sum(vw_distribution)

    clean_vw_distribution = delete_stop_words(vw_distribution)
    clean_vocabulary_pct = len(clean_vw_distribution) / vocabulary_size * 100
    clean_vw_pct = sum(clean_vw_distribution) / vw_total * 100

    print('Original size vocabulary:', vocabulary_size)
    print('Clean size vocabulary: ', "{:.2f}".format(clean_vocabulary_pct) + '%')

    print('Number of visual words before the reduction:', vw_total)
    print('Visual words after the reduction: ', "{:.2f}".format(clean_vw_pct) + '%')
    return features_df[features_df['visual_word_id'].isin(clean_vw_distribution.index)]


def plot_distribution(desc_distribution, name):
    fig = plt.figure(figsize=(15, 10))
    desc_distribution.hist(bins=100)
    plt.xlabel('Frecuencia en el conjunto de datos')
    plt.ylabel('Palabras visuales')
    fig.savefig("/data/images/plots/hist_desc_" + name + ".jpg", dpi=256, bbox_inches='tight')


def main():
    with open(args.cluster + '.csv', 'r') as file:
        features_df = pd.read_csv(file)
    if args.drop_outliers:
        features_df = reduce_vocabulary(features_df)
    gen_document(features_df, args.use_size)


if __name__ == '__main__':
    main()
