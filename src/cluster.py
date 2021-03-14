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
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('descriptors',
  help = 'Ruta del archivo generado por el extractor')
parser.add_argument('cluster',
  help = 'Ruta del archivo a generar')
parser.add_argument('Nclusters', help = 'Número de clusters', type=int)
args = parser.parse_args()

def main(args):
    with open(args.descriptors, 'rb') as pickle_file:
        params = pickle.load(pickle_file) #Parámetros de extración
        desc_imgs = pickle.load(pickle_file) #Lista de descriptores
        b = [item['descriptors'] for item in desc_imgs] #"Vectores descriptores"
        c = [item['name_img'] for item in desc_imgs] #"Nombre de la imagén "
        #Longitud de cada vector descriptor
        hi=[]
        for i in range(0,len(b),1):
            hi.append(len(b[i]))
        #Lista de descripores con sus n entradas
        descriptores = b
        desc = np.array(descriptores)
        J=[]
        for D in desc:
            for i in D:
                J.append(i)
        #Centroides y etiquetas de cada vector descriptor
        print("\nDatos cargados correctamente, iniciando clusterización...\n")
        kmeans = MiniBatchKMeans(n_clusters=args.Nclusters, init='k-means++',
          n_init=10, max_iter=300, tol=0.0001 ).fit(J)
        centroids = kmeans.cluster_centers_
        etiquetas = kmeans.labels_
        la = kmeans.predict(J)
        #Suma de las longitudes de los vectores
        laSuma = 0
        nu=[]
        for i in hi:
            laSuma = laSuma + i
            nu.append(laSuma)
        #Se le agrega el 0 a la lista
        nu.insert(0,0)
        #Etiquetas agrupadas por por vector descriptor (listas de etiquetas)
        lab=[]
        for i in range(0,len(nu)-1):
            lab.append(etiquetas[nu[i]:nu[i+1]])
        # Cambiamos el tamaño del arreglo anterior
        my_array= np.array(lab)
        poo=my_array.reshape((len(lab),1))
        # Agregamos los nombres al arreglo
        cool=np.insert(poo, poo.shape[1], np.array((c)), 1)
        #Formato en pandas
        labels=pd.DataFrame(cool)
        #Se guarda en un archivo pickle
        with open(args.cluster, 'wb') as pickle_file:
            pickle.dump(labels, pickle_file)
main(args)
