#!/usr/bin/env python
# coding: utf-8

import sys
import os.path as path 

modulos_path = path.abspath('../minIA')
if modulos_path not in sys.path:
    sys.path.append(modulos_path)

import cv2 as cv
from matplotlib import pyplot as plt
from math import log
import numpy as np
from utiles import lectura_img
from tqdm import tqdm
import argparse

#Argumentos
parser = argparse.ArgumentParser(
        description='''Aplica una mascara a cada una de las imagenes del
        directorio original y las guarda en el directorio destino''')
parser.add_argument("th", 
        help = "Threshold o umbral", type=int)
parser.add_argument("kernel",
        help="Tamaño del kernel del filtro de suavizado", type=int)
parser.add_argument("dir_img",
        help = 'Ruta del directorio de imágenes.')
parser.add_argument("dir_output",
        help = 'Ruta del directorio de salida de imágenes con máscara.')

#Introduciendo todos los argumentos. 
args = parser.parse_args()

#Obtención de imágenes
path_images = lectura_img(args.dir_img)

#Ciclo de implementación de máscara
for image in tqdm(path_images):
    #Leemos la imagen a color y después la convertimos a escala de grises
    img_c = cv.imread(image, cv.IMREAD_COLOR)
    img_gris = cv.cvtColor(img_c, cv.COLOR_BGR2GRAY)
    #Filtro de suavizado para homogenizar ruido de fondo
    mask = cv.blur(img_gris, (args.kernel, args.kernel))
    #Creación de la máscara
    mask[mask <= args.th] = 0
    mask[mask > 0] = 1
    #Aplicación de la máscara
    img_filter = cv.bitwise_and(img_c, img_c, mask = mask)
    #Extraemos el nombre de la img del path
    base_nom = path.basename(image)
    nom_img = path.splitext(base_nom)[0]
    #Guardamos la imagen
    cv.imwrite(args.dir_output + "/" + nom_img + ".jpeg", img_filter)
    
