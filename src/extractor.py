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

modulos_path = path.abspath('../minIA)
if modulos_path not in sys.path:
    sys.path.append(modulos_path)

import utiles.lectura_img
import argparse
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("extr", choices=['SIFT', 'SURF', 'DELF'])
#se pasa el nombre del directorio dentro de la carpeta images EJ: GZoom/Grupo1
args = parser.parse_args()

'''
Se planea que todos los extractores implementen esta clase, para que el código
principal (main) no tenga modificaciones y funcione igual independientemente del
método de extración.
'''
class Extractor(object):
    def calculoDescriptores(self, imagen):
        raise NotImplementedError('todas las subclases deben sobrescribir')'



class Sift(Extractor):
    def __init__(self, gray):
        self.sift = cv.SIFT_create()
    def calculoDescriptores(self, imagen):
        self.kp, self.des = self.sift.detectAndCompute(imagen,None)


#Definición del tipo de extractor
if args.extr == 'SIFT':
    extractor = Sift()
elif args.extr == 'SURF':
    #extractor = Surf()
    pass
else
    #extractor = Delf()
    pass


#main
path_images = lectura_img('Grupo1')
descriptores = list()
for imagen in path_images:
    img = cv.imread(imagen)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #NOTA: si los otros detectores no requieren imagen en escala de grises, este paso es exclusivo de SIFT y debería ir en su método
    descriptores.append(extractor.calculoDescriptores(gray))
    #Escribir la lista en archivo de salida.
