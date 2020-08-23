#!/usr/bin/env python
# coding: utf-8

from os import path, scandir

'''
Obtiene una lista con la ruta de cada archivo en el directorio especificado,
si la ruta no existe retornará una lista vacía. ¡El directorio de búsqueda NO
debe contener subdirectorios!
Parámetros:
    Dirección absoluta o relativa de búsqueda
'''
def lectura_img(name_dir):
    path_img = path.join(path.abspath('..'), name_dir)
    if path.exists(path_img):
        return [archivo.path for archivo in scandir(path_img)]
    elif path.exists(name_dir):
        return [archivo.path for archivo in scandir(name_dir)]
    else:
        print("\nLa ruta especificada no existe:")
        print(path_img+"\n")
        return list()
