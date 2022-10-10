#!/usr/bin/env python
# coding: utf-8
"""
Extract images features from image directory
"""
import os.path as path

import numpy as np
import pandas as pd

from minIA.utiles import lectura_img
from minIA.models import Delf, Sift, Surf
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("extr",
                    help='Extractor', choices=['SIFT', 'SURF', 'DELF'])
parser.add_argument("dir",
                    help='Image directory path')
parser.add_argument("dir_output",
                    help='Output file path')
parser.add_argument("-auto",
                    help='Set automatic params for SIFT and SURF', action="store_true")

parser.add_argument('-median_filter',
                    help='Median filter', default=False, type=bool)

parser.add_argument("-threshold",
                    help='SURF parameter', default=100, type=int)
parser.add_argument("-nOctaves",
                    help='SURF parameter', default=4, type=int)
parser.add_argument("-nOctaveLayers",
                    help='SURF parameter y SIFT', default=3, type=int)
parser.add_argument("-extended",
                    help='SURF parameter', default=False, type=bool)
parser.add_argument("-upright",
                    help='SURF parameter', default=True, type=bool)

parser.add_argument("-n_features",
                    help="SIFT parameter", default=0, type=int)
parser.add_argument("-contrastThreshold",
                    help="SIFT parameter", default=0.04, type=float)
parser.add_argument("-edgeThreshold",
                    help="SIFT parameter", default=10, type=float)
parser.add_argument("-sigma",
                    help="SIFT parameter", default=1.6, type=float)

parser.add_argument("-delf_configuration",
                    help="DELF file configuration",
                    default='/data/config/delf_config_galaxy.pbtxt')

args = parser.parse_args()
line = 'index,location,size,image_name\n'
fmt = '%d'

# Extractor definition
if args.extr == 'SIFT':
    extractor = Sift(args.auto, args.n_features, args.nOctaveLayers,
                     args.contrastThreshold, args.edgeThreshold, args.sigma)
elif args.extr == 'SURF':
    extractor = Surf(args.auto, args.threshold, args.nOctaves,
                     args.nOctaveLayers, args.extended, args.upright)
else:
    extractor = Delf(args.delf_configuration)
    line = 'index,location,size,score,image_name\n'
    fmt = '%.24f'

if args.median_filter:
    extractor.filter = True


def main():
    images_paths = lectura_img(args.dir)
    path_file = path.abspath(args.dir_output + '_' + args.extr)
    descriptors_file = open(path_file + '.txt', 'wb')
    features_file = open(path_file + '.csv', 'w')
    features_file.write(line)
    for image_path in tqdm(images_paths):
        nom_img = path.split(image_path)[1][:-4]
        img_features, img_descriptors = extractor.get_features(image_path)
        if img_features is not None:
            img_features_df = pd.DataFrame(img_features)
            img_features_df['name_img'] = nom_img
            np.savetxt(descriptors_file, img_descriptors, fmt=fmt)
            img_features_df.to_csv(features_file, mode='a', header=False)
    print('Features extracted! ' + args.extr)


if __name__ == '__main__':
    main()
