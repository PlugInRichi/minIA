#!/usr/bin/env python
# coding: utf-8

from os import path, scandir

#Obtiene una lista con la ruta de cada archivo en el directorio
#images del proyecto
def lectura_img():
    path_img = path.join(path.abspath('..'),'images')
    if path.exists(path_img):
        return [archivo.path for archivo in scandir(path_img)]
    else:
        print("La ruta especificada no existe")
        return list()
