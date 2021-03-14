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
from collections import Counter
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("magnitude", help='Selecciona la magnitud asociada a los índices de la imagen',
  choices=['FRECUENCY', 'WEIGHT'])
parser.add_argument("original", help='Ruta del archivo de descriptores originales')
parser.add_argument("cluster", help='Ruta del archivo de descriptores clusterizados')
parser.add_argument("document_name", help='Ruta y nombre del archivo a crear')
args = parser.parse_args()



def genDocumentFrecuency(desc_imgs,images_descr):
    """
    En cada imagen, toma cada descriptor de la imagen y cuenta el número de
    veces que aparece en él.
    """
    with open(args.document_name, 'w') as file:

        img = 0
        for descr in tqdm(images_descr[0]):
            cnt_caract = Counter(descr)
            row = str(len(cnt_caract))
            for car in sorted(cnt_caract.keys()):
                row+= ' '+ str(car) +':'+str(cnt_caract[car])
            file.write(row+'\n')
            img+=1


def genDocumentWeight(desc_imgs,images_descr):
    """
    En cada imagen, toma cada descriptor de la imagen y mide el tamaño asociado
    a cada uno de ellos, si se repiten los descriptores suma ambos tamaños.
    """
    with open(args.document_name, 'w') as file:
        #Obtiene lista de KP para cada descriptor
        img_keypoints = [x['keypoints'] for x in desc_imgs]
        img_centroides = list(images_descr[0])

        for centroides, keypoint in tqdm(zip(img_centroides, img_keypoints),
          total=len(img_centroides)):
            #Diccionarios con Centroide, peso en 0
            img = (dict( (cent, 0) for cent in set(centroides)))
            for cent, kp in  zip(centroides, keypoint):
                img[cent] += round(kp[2]) #Incrementa el tamaño
            row = str(len(img))+" "+str(img).replace(", "," ").replace(": ",":")[1:-1]
            file.write(row+'\n')



def main(args):
    with open(args.original, 'rb') as file_original:
        args_info = pickle.load(file_original)
        desc_imgs = pickle.load(file_original)
    with open(args.cluster, 'rb') as file_cluster:
        images_descr = pickle.load(file_cluster)
    #Verificación de simetría, no ejecutar si no son simétricos
    if (len(images_descr) != len(desc_imgs)):
        print("\nLos índices están desfasados, verificar la cardinalidad")
        print("Cluster: " + str(len(images_descr)))
        print("Original: " + str(len(desc_imgs))+"\n")
        exit()
    print("\nArchivos cargados correctamente, generando archivo...")
    if args.magnitude == 'FRECUENCY':
        genDocumentFrecuency(desc_imgs,images_descr)
    else:
        genDocumentWeight(desc_imgs,images_descr)

main(args)
