#!/usr/bin/env python
# coding: utf-8

from os import path, scandir

#Obtiene una lista con la ruta de cada archivo en el directorio especificado
def lectura_img(name_dir):
    path_img = path.join(path.abspath('..'),'images\\'+ name_dir)
    if path.exists(path_img):
        return [archivo.path for archivo in scandir(path_img)]
    else:
        print("La ruta especificada no existe:")
        print(path_img)
        return list()
