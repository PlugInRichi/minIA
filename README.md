# minIA
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.2-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
![Maintaner](https://img.shields.io/badge/OpenCV_contrib_python-3.4.2.16-blue)

_El proyecto tiene como finalidad encontrar características dentro de una galería de imágenes que permita identificar los patrones representen estructuras de las galaxias. Haciendo uso de un vocabulario visual se extraen los objetos visuales por medio de Sampled-MinHashing_

## Comenzando 🚀

### Instalación en linux

Crear ambiente

```
virtualenv -p python3.6 env
```

Activar el ambiente

```
source env/bin/activate
```

Instalar librerias 

```
pip install -r requirements.txt
```

Adicionalmente será necesario instalar **Sampled-MinHashing**, para ello puede seguir las intrucciones descritas aquí: [https://github.com/gibranfp/Sampled-MinHashing]

# Ejecución :joystick:

## Entrenamiento del modelo especializado en detección de características
1. Creación de dataset
2. Reformating dataset
3. Train
4. Export Model

### Creación de dataset

### Reformating dataset

### Train

### Export Model

## Descubrimiento de estructuras visuales
1. [Extracción de características](#Extracción-de-características)
2. [Clusterización](#Clusterización)
3. [Minado de estructuras](#Minado-de-estructuras)
4. [Visualización de estructuras](#Visualización-de-estructuras)

### Extracción de características 

El script encargado de extraer los descriptores es _extractor.py_ para su ejecución es obligatorio la especificación de tres parámetros:
1. Tipo de extractor
2. Ruta Absoluta o Relativa de la carpeta de imágenes
3. Ruta Absoluta o Relativa del archivo a generar

La siguiente ejecución creará un archivo Pickle con el nombre _images_descriptors_
> extractor.py SIFT /images_dataset /test/images_descriptors


### Clusterización
El proceso de minado requiere de un vocabulario, utilizando los descriptores generados del paso anterior se realiza un clusterizado con la finalidad de estandarizar nuestro vocabulario, al final de este proceso se obtendrá un nuevo archivo que contendrá los un índice que representa el cluster al que fue asociado cada descriptor. 

El script encargado de extraer los descriptores es cluster.py_ para su ejecución es obligatorio la especificación de tres parámetros:
1. Ruta Absoluta o Relativa del archivo generado por el paso anterior
2. Ruta Absoluta o Relativa del archivo a generar
3. Número de cluster (tamaño de vocabulario final)

La siguiente ejecución creará un archivo Pickle con el nombre _images_clusters_ utilizando 2000 clusters
> cluster.py /test/images_descriptors /test/images_clusters 2000

### Minado de estructuras 
Utilizando el archivo generado de la clusterización y el generado de la extracción se crea el doumento de entrada para SHM.

1. Magnitud asociada a los índices de la imagen (Tamaño o frecuencia)
2. Ruta Absoluta o Relativa del archivo generado por el extractor
3. Ruta Absoluta o Relativa del archivo generado por el cluster
4. Ruta Absoluta o Relativa del documento a generar

La siguiente ejecución creará un documento con el nombre _clusters_per_images_ midiendo el tamaño con el que aparecen dentro de la imagen
> SMHdocument.py SIZE /test/images_descriptors /test/images_clusters  /test/clusters_per_images

EJECUTAR SMH...

### Visualización de estructuras 

