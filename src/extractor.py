#!/usr/bin/env python
# coding: utf-8

'''
Entradas rutas de directorio con imagenes
Salida guardar en un archivo
Seleccionar extractor de caract por comandos (argparse)
Controlar parámetros extraccion
Control de formato a la entrada (tamaño y normalización)
'''
import sys
import os.path as path

modulos_path = path.abspath('../minIA')
if modulos_path not in sys.path:
    sys.path.append(modulos_path)

from utiles import lectura_img
import argparse
import numpy as np
import cv2 as cv
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("extr", help='Extractor', choices=['SIFT', 'SURF', 'DELF'])
parser.add_argument("dir", help='Ruta del directorio de imagenes')
parser.add_argument("nArch", help='Nombre del archivo de salida')

parser.add_argument("-threshold", help='Parametro de SURF', default=100, type=int)
parser.add_argument("-nOctaves", help='Parametro de SURF', default=4, type=int)
parser.add_argument("-nOctaveLayers", help='Parametro de SURF y SIFT', default=3, type=int)
parser.add_argument("-extended", help='Parametro de SURF', default=False, type=bool)
parser.add_argument("-upright", help='Parametro de SURF', default=True, type=bool)

#Se dejan los valores establecidos por el paper, aunque pueden ser calculadas en automático
parser.add_argument("-nfeatures", help="Parametro de SIFT", default=0, type=int)
parser.add_argument("-contrastThreshold", help="Parametro de SIFT", default=0.04, type=float)
parser.add_argument("-edgeThreshold", help="Parametro de SIFT", default=10, type=float)
parser.add_argument("-sigma", help="Parametro de SIFT", default=1.6, type=float)

args = parser.parse_args()

'''
Se planea que todos los extractores implementen esta clase, para que el código
principal (main) no tenga modificaciones y funcione igual independientemente del
método de extración.
'''
class Extractor(object):
    def calculoDescriptores(self, imagen):
        raise NotImplementedError('todas las subclases deben sobrescribir')


class Sift(Extractor):
    def __init__(self, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):
        self.sift = cv.xfeatures2d_SIFT.create(nfeatures, nOctaveLayers,
        contrastThreshold, edgeThreshold, sigma)
    def calculoDescriptores(self, imagen):
        return  self.sift.detectAndCompute(imagen,None)

class Surf(Extractor):
    def __init__(self, threshold, nOctaves, nOctaveLayers, extended, upright):
        self.surf = cv.xfeatures2d.SURF_create(threshold, nOctaves, nOctaveLayers,
        extended, upright)
    def calculoDescriptores(self, imagen):
        return  self.surf.detectAndCompute(imagen,None)



#Definición del tipo de extractor
if args.extr == 'SIFT':
    extractor = Sift(args.nfeatures, args.nOctaveLayers, args.contrastThreshold,
    args.edgeThreshold, args.sigma)
elif args.extr == 'SURF':
    extractor = Surf(args.threshold, args.nOctaves, args.nOctaveLayers,
    args.extended, args.upright)
else:
    #extractor = Delf()
    pass


#main
path_images = lectura_img(args.dir)
pickle_file = open(args.nArch+'_'+args.extr+'.pickle', 'wb')
descriptores = list()

for imagen in path_images:
    img = cv.imread(imagen, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = extractor.calculoDescriptores(img)
    descriptores.append([cv.KeyPoint_convert(keypoints),descriptors])
pickle.dump(args, pickle_file)
pickle.dump(descriptores, pickle_file)
print('¡Listo! ' + args.extr)
