#!/usr/bin/env python
# coding: utf-8

"""
Genera el vocabulario de palabras visuales

Entrada:
    Archivo de descriptores originales (generado por el extractor)
    Número de clusters a generar
Salida:
    Archivo de descriptores clusterizados
"""


import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('descriptors',
  help = 'Ruta del archivo generado por el extractor')
parser.add_argument('cluster',
  help = 'Ruta del archivo a generar')
parser.add_argument('Nclusters',
  help = 'Número de clusters', type=int)
args = parser.parse_args()

def main(args=args):
    columns_name = ['index', 'location', 'size', 'image_name']
    indexes_name = ['index', 'image_name']
    with open(args.descriptors+'.csv', 'r') as file:
        features_df = pd.read_csv(file, names=columns_name, index_col=indexes_name)
    descriptors = np.loadtxt(args.descriptors+'.txt')

    print("\nDatos cargados correctamente, iniciando clusterización...\n")
    kmeans = MiniBatchKMeans(n_clusters=args.Nclusters,
                             init='k-means++',
                             n_init=1,
                             batch_size=4096,
                             verbose=1,
                             max_iter=10,
                             tol=0.000001).fit(descriptors)

    features_df['descriptor_id'] = kmeans.labels_.astype(int).tolist()
    features_df.to_csv(args.cluster+'.csv')

if __name__ == '__main__':
  main()
