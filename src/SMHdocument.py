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

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("magnitude",
                    help='Selecciona la magnitud asociada a los índices de la imagen',
                    choices=['FRECUENCY', 'SIZE'])
parser.add_argument("original",
                    help='Ruta del archivo de descriptores generado por el extractor')
parser.add_argument("cluster",
                    help='Ruta del archivo de descriptores generado por el cluster')
parser.add_argument("document_name",
                    help='Ruta y nombre del documento a generar')
parser.add_argument('-drop_outliers',
                    help='Elimina los descriptores anormales', default=False, action="store_true")
args = parser.parse_args()


def genDocumentFrecuency(descriptors_df):
    with open(args.document_name, 'w') as file:
        img = 0
        for name, group in descriptors_df.groupby(['file_id']):
            cnt_caract = Counter(group['descriptors'])
            row = str(len(cnt_caract))
            for car in sorted(cnt_caract.keys()):
                row += ' ' + str(car) + ':' + str(cnt_caract[car])
            file.write(row + '\n')
            img += 1

def genDocumentSize(desc_imgs, images_descr, selected_index_desc):
    """
    En cada imagen, toma cada descriptor de la imagen y mide el tamaño asociado
    a cada uno de ellos, si se repiten los descriptores suma ambos tamaños.
    """
    with open(args.document_name, 'w') as file:
        # Obtiene lista de KP para cada descriptor
        img_keypoints = [x['keypoints'] for x in desc_imgs]
        img_centroides = list(images_descr[0])  # Poner lista de centroides por imagen

        for centroides, keypoint in tqdm(zip(img_centroides, img_keypoints), total=len(img_centroides)):
            # Diccionarios con Centroide, peso en 0
            # centroides
            img = (dict((cent, 0) for cent in set(centroides)))
            for cent, kp in zip(centroides, keypoint):
                if cent in selected_index_desc:
                    img[cent] += round(kp[2])  # Incrementa el tamaño
            data = str(len(img)) + " " + str(img)
            row = data.replace(", ", " ").replace(": ", ":")[1:-1]
            file.write(row + '\n')


def get_data(full_desc_path, cluster_desc_path):
    with open(full_desc_path, 'rb') as full_desc_file:
        args_info = pickle.load(full_desc_file)
        full_desc = pickle.load(full_desc_file)
    with open(cluster_desc_path, 'rb') as cluster_desc_file:
        cluster_desc = pickle.load(cluster_desc_file)
    # Verificación de simetría, cada entrada es una imagen
    if (len(cluster_desc) != len(full_desc)):
        print("\nLos índices están desfasados, verificar la cardinalidad")
        print("Cluster: " + str(len(cluster_desc)))
        print("Original: " + str(len(full_desc)) + "\n")
        exit()
    print("\nArchivos cargados correctamente, generando archivo...")
    return full_desc, cluster_desc


def drop_outliers(desc_distribution):
    plot_distribution(desc_distribution, 'original')
    mean_desc = desc_distribution.mean()
    std_desc = desc_distribution.std()
    low_limit = mean_desc - 2 * std_desc
    upper_limit = mean_desc + 2 * std_desc
    desc_distribution = desc_distribution[desc_distribution > low_limit]
    desc_distribution = desc_distribution[desc_distribution < upper_limit]
    plot_distribution(desc_distribution, '2sigma')
    return desc_distribution


def plot_distribution(desc_distribution, name):
    fig = plt.figure(figsize=(15, 10))
    desc_distribution.hist(bins=20)
    plt.xlabel('Número de descriptores')
    plt.ylabel('Índices')
    fig.savefig("/data/images/plots/hist_desc" + name + ".jpg", dpi=256, bbox_inches='tight')


def refactor_df(cluster_desc):
    labels_desc = cluster_desc.apply(lambda x: pd.Series(x[0]), axis=1).stack().reset_index(level=1, drop=True)
    labels_desc.name = 'descriptors'
    descriptors_df = cluster_desc.drop(0, axis=1).join(labels_desc)
    descriptors_df.columns = ['file_id', 'descriptors']
    descriptors_df['descriptors'] = descriptors_df['descriptors'].astype(np.int)
    return descriptors_df


def main(args):
    full_desc, cluster_desc_df = get_data(args.original, args.cluster)
    descriptors_df = refactor_df(cluster_desc_df)
    desc_distribution = descriptors_df['descriptors'].value_counts()
    all_fetures = sum(desc_distribution)
    print('Total de palabras visuales:', len(desc_distribution))
    if args.drop_outliers:
        desc_distribution = drop_outliers(desc_distribution)
        print('Palabras visuales reducidas:', len(desc_distribution),
              'características utilizadas:', sum(desc_distribution)/all_fetures * 100)
        descriptors_df = descriptors_df[descriptors_df['descriptors'].isin(desc_distribution.index)]
    if args.magnitude == 'FRECUENCY':
        genDocumentFrecuency(descriptors_df)
    else:
        genDocumentSize(full_desc, cluster_desc_df, set(desc_distribution.index))


if __name__ == '__main__':
    main(args)
