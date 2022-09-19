#!/usr/bin/env python
# coding: utf-8

from minIA.utiles import lectura_img
from minIA.imageProcessing import filter_images
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source_dir", help='Ruta del directorio de imagenes')
parser.add_argument("dest_dir", help='Ruta del directorio de imagenes filtradas')


def main(args=parser.parse_args()):
    print('Filtering images...')
    path_names = lectura_img(args.source_dir)
    image_names = [im_path.split('/')[-1][:-4] for im_path in path_names]
    filter_images(image_names, args.source_dir, args.dest_dir)


if __name__ == '__main__':
    main()
