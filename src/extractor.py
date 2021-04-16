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
from tqdm import tqdm
import argparse
import numpy as np
import cv2 as cv
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("extr", help='Extractor', choices=['SIFT', 'SURF', 'DELF'])
parser.add_argument("dir", help='Ruta del directorio de imagenes')
parser.add_argument("dir_output", help='Ruta del archivo de salida')
parser.add_argument("-auto", help='Cálculo de parámetros automático', action="store_true")

parser.add_argument('-median_filter', help='Filtro de mediana', default=False, type=bool)
parser.add_argument('-median_value', help='Valores Impares', default=15, type=int)

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
        #      [[ [x, y, size] , [descriptor], [nombreArch] ] ....[]   ]


class Sift(Extractor):
    def __init__(self, auto, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):
        if auto:
            self.sift = cv.xfeatures2d_SIFT.create()
        else:
            self.sift = cv.xfeatures2d_SIFT.create(nfeatures, nOctaveLayers,
            contrastThreshold, edgeThreshold, sigma)

    def calculoDescriptores(self, imagen):
        kps, descs = self.sift.detectAndCompute(imagen,None)
        keypoints = list()
        for kp in kps:
            keypoints.append([kp.pt[0], kp.pt[1], kp.size])
        return  {'keypoints': keypoints, 'descriptors':descs}


class Surf(Extractor):
    def __init__(self, auto, threshold, nOctaves, nOctaveLayers, extended, upright):
        if auto:
            self.surf = cv.xfeatures2d.SURF_create()
        else:
            self.surf = cv.xfeatures2d.SURF_create(threshold, nOctaves, nOctaveLayers,
            extended, upright)

    def calculoDescriptores(self, imagen):
        kps, descs = self.surf.detectAndCompute(imagen,None)
        keypoints = list()
        for kp in kps:
            keypoints.append([kp.pt[0], kp.pt[1], kp.size])
        return  {'keypoints': keypoints, 'descriptors':descs}



#Definición del tipo de extractor
if args.extr == 'SIFT':
    extractor = Sift(args.auto, args.nfeatures, args.nOctaveLayers, args.contrastThreshold,
    args.edgeThreshold, args.sigma)
elif args.extr == 'SURF':
    extractor = Surf(args.auto, args.threshold, args.nOctaves, args.nOctaveLayers,
    args.extended, args.upright)
else:
    #extractor = Delf()
    pass


#main
'''
Exporta una lista (archivo pickle) que contiene los keypoints,
descriptores y nombre del archivo para cada imagen encontrada en  el directorio
'''
path_images = lectura_img(args.dir)
path_pickle = path.abspath(args.dir_output+'_'+args.extr+'.pickle')
descriptores = list()
pickle_file = open(path_pickle, 'wb')
for imagen in tqdm(path_images):
    
    if args.median_filter == True:
        img = cv.imread(imagen, cv.COLOR_HSV2BGR)
        image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img= cv.medianBlur(image_gray, args.median_value)
    else:
        img = cv.imread(imagen, cv.COLOR_BGR2GRAY)
    
    nom_img = path.split(imagen)[1]
    descs_img = extractor.calculoDescriptores(img)
    descs_img['name_img'] = nom_img
    descriptores.append(descs_img)
   
Descriptores = [item['descriptors'] for item in descriptores] #"Vectores descriptores"
Keypoints = [item['keypoints'] for item in descriptores] #"Posicion del vector descriptor"
Nombre_Img = [item['name_img'] for item in descriptores] #"Nombre de la imagén "

Descriptores2 = list()
Keypoints2 = list()
Nombre_Img2 = list()

Descriptores3 = list()
Keypoints3 = list()
Nombre_Img3 = list()

for i in Keypoints:
    if len(i)==0:
        Keypoints3.append(i)
    else:
        Keypoints2.append(i)

for j,k in zip(Descriptores,Nombre_Img):
    if j is None:
        Descriptores3.append(j)
        Nombre_Img3.append(k)
    else:
        Descriptores2.append(j)
        Nombre_Img2.append(k) 
        
descriptores2 = list()
for i in np.arange(0,len(Keypoints2),1):
    jun = dict(keypoints = Keypoints2[i],descriptors = Descriptores2[i],name_img = Nombre_Img2[i])
    descriptores2.append(jun)
    
pickle.dump(args, pickle_file)
pickle.dump(descriptores2, pickle_file)
print('¡Listo! ' + args.extr)
