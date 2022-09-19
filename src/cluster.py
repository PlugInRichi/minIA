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
    with open(args.descriptors, 'rb') as pickle_file:
        _ = pickle.load(pickle_file) #Parámetros de extración
        data = pickle.load(pickle_file) #Lista de descriptores por imagen

    list_descs = [img['descriptors'] for img in data] #"Vectores descriptores"
    names = [img['name_img'] for img in data] #"Nombres"
    all_descs = [desc for descs_img in list_descs for desc in descs_img]

    #clusterización
    print("\nDatos cargados correctamente, iniciando clusterización...\n")
    kmeans = MiniBatchKMeans(n_clusters=args.Nclusters, init='k-means++',
      n_init=25, batch_size=4096, verbose=1, max_iter=10, tol=0.000001).fit(all_descs)

    etiquetas = kmeans.labels_
    long_desc = np.array([len(descs_img) for descs_img in list_descs])
    indexes = np.cumsum(long_desc)

    #Segmentación de todos los descriptores por cada imagen
    list_etiq = np.split(etiquetas, indexes[:-1])
    list_etiq = np.array(list_etiq, dtype=np.ndarray).reshape(-1,1)
    data = np.insert(list_etiq, 1, np.array((names)), axis=1)

    with open(args.cluster, 'wb') as pickle_file:
        pickle.dump(pd.DataFrame(data), pickle_file)

if __name__ == '__main__':
  main()
