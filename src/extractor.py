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

parser = argparse.ArgumentParser()
parser.add_argument("extr", help='Extractor', choices=['SIFT', 'SURF', 'DELF'])
parser.add_argument("dir", help='Nombre del directorio de imagenes')
parser.add_argument("-threshold", help='Parametro de SURF', type=int)
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
    def __init__(self):
        #self.sift = cv.SIFT_create()
        self.sift = cv.xfeatures2d_SIFT.create()
    def calculoDescriptores(self, imagen):
        return  self.sift.detectAndCompute(imagen,None)

class Surf(Extractor):
    def __init__(self, threshold):
        self.surf = cv.xfeatures2d.SURF_create(threshold)
    def calculoDescriptores(self, imagen):
        return  self.surf.detectAndCompute(imagen,None)



#Definición del tipo de extractor
if args.extr == 'SIFT':
    extractor = Sift()
elif args.extr == 'SURF' and args.threshold is not None:
    extractor = Surf(args.threshold)
else:
    #extractor = Delf()
    pass


#main
path_images = lectura_img(args.dir)
descriptores = list()
for imagen in path_images:
    img = cv.imread(imagen, cv.COLOR_BGR2GRAY)
    descriptores.append(extractor.calculoDescriptores(img))
    #Escribir la lista en archivo de salida.
print('¡Listo! ' + args.extr)
