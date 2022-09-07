#!/usr/bin/env python
# coding: utf-8
"""
Exporta una lista (archivo pickle) que contiene los keypoints, descriptores
y nombre del archivo para cada imagen encontrada en el directorio.
"""
import os.path as path

from minIA.utiles import lectura_img
from minIA.models import Delf, Sift, Surf
from tqdm import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("extr",
    help='Extractor', choices=['SIFT', 'SURF', 'DELF'])
parser.add_argument("dir",
    help='Ruta del directorio de imagenes')
parser.add_argument("dir_output",
    help='Ruta del archivo de salida')
parser.add_argument("-auto",
    help='Cálculo de parámetros automático', action="store_true")

parser.add_argument('-median_filter',
    help='Filtro de mediana', default=False, type=bool)
parser.add_argument('-median_value',
    help='Valores Impares', default=15, type=int)

parser.add_argument("-threshold",
    help='Parametro de SURF', default=100, type=int)
parser.add_argument("-nOctaves",
    help='Parametro de SURF', default=4, type=int)
parser.add_argument("-nOctaveLayers",
    help='Parametro de SURF y SIFT', default=3, type=int)
parser.add_argument("-extended",
    help='Parametro de SURF', default=False, type=bool)
parser.add_argument("-upright",
    help='Parametro de SURF', default=True, type=bool)


parser.add_argument("-nfeatures",
    help="Parametro de SIFT", default=0, type=int)
parser.add_argument("-contrastThreshold",
    help="Parametro de SIFT", default=0.04, type=float)
parser.add_argument("-edgeThreshold",
    help="Parametro de SIFT", default=10, type=float)
parser.add_argument("-sigma",
    help="Parametro de SIFT", default=1.6, type=float)

parser.add_argument("-delf_configuration",
    help="Archivo de configuración delf", default='/data/config/delf_config_galaxy.pbtxt')

args = parser.parse_args()

#Definición del tipo de extractor
if args.extr == 'SIFT':
    extractor = Sift(args.auto, args.nfeatures, args.nOctaveLayers,
    args.contrastThreshold, args.edgeThreshold, args.sigma)
elif args.extr == 'SURF':
    extractor = Surf(args.auto, args.threshold, args.nOctaves,
    args.nOctaveLayers, args.extended, args.upright)
else:
    extractor = Delf(args.delf_configuration)

if args.median_filter:
    extractor.filter = True


def main(args=args):
    images_paths = lectura_img(args.dir)
    path_pickle = path.abspath(args.dir_output+'_'+args.extr+'.pickle')
    descriptors= list()
    pickle_file = open(path_pickle, 'wb')
    for image_path in tqdm(images_paths):
        nom_img = path.split(image_path)[1]
        descs_img = extractor.get_features(image_path)
        if descs_img['descriptors'] is not None:
            descs_img['name_img'] = nom_img
            descriptors.append(descs_img)

    pickle.dump(args, pickle_file)
    pickle.dump(descriptors, pickle_file)
    print('¡Listo! ' + args.extr)

if __name__ == '__main__':
    main()
