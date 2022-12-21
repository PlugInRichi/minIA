# minIA
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.2-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
![Maintaner](https://img.shields.io/badge/OpenCV_contrib_python-3.4.2.16-blue)

_**minIA** is an unsupervised learning methiodology for discovering patterns in astronomical images (although it can be applied to any other image collection). We use a customized DELF model to extract simple features from images through a bag of words model we represent each image in the collection to mine patterns using the ![Sampled-MinHashing](https://github.com/gibranfp/Sampled-MinHashing) technique._

## Comenzando 🚀
El proyecto completo puede ejecutarse desde un contenedor, las intrucciones para ejecutarlo correctamente pueden ser encontradas [aquí](install).


El uso de **Sampled-MinHashing** requiere de una instalación que no se encuentra en el contenedor, para esto hay que seguir las intruciones en [Sampled-MinHashing](https://github.com/gibranfp/Sampled-MinHashing) repositorio.

# Ejecución :joystick:

## Entrenamiento del modelo especializado en detección de características
1. Dataset creation
2. Formating dataset
3. Train
4. Export Model

### Creación de dataset
Para configurar la creación del dataset de entrenamiento para el modelo neuronal es necesario especificar los parámetros en un archivo de configuración:
```bash
# data/config/dataset_config.yml
    galaxyZoo2_path:  /data/images/gz2_hart16.csv
    map_images_path:  /data/images/gz2_filename_mapping.csv
    train_dataset_path: /data/images/gz2_train_dataset_5000
    images_dir_path: /data/images/images_gz2
    images_out_dir_path: /data/images/images_gz2
    class_size: 5000
```
Después simplemente ejecutar el script
```
python3 createDataSet.py
```
### Reformating dataset
```
python3 custom_delf/build_galaxy_image_dataset.py \
  --train_clean_csv_path=/data/images/gz2_train_dataset_5000_with_filter.csv \
  --train_directory=/data/images/images_gz2/  \
  --output_directory=/data/tf_records/v5-full_merge \
  --num_shards=64 \
  --validation_split_size=0.2
```
*** Nota: para entrenar sobre cualquier otro dataset se requiere tener un
formato igual al de 'GZ_dataset.csv' (respetar número de espacios y saltos de
línea, el nombre de las imagenes se da sin formato y la extensión debe ser JPG)

```
Categoria_Encabezado,Nombre_encabezado
Categoria_ID,nombre_imagen_1 nombre_imagen_2 ...
Categoria_ID,nombre_imagen_2 nombre_imagen_5 ...
```
*** Nota: Cada imagen pertenece solo a una categoría
*** Nota: Los nombres de los encabezados deben de coincidir con los discritos en el script

### Train
```
python3 custom_delf/train.py \
    --train_file_pattern=/data/tf_records/v5-full_merge/train* \
    --validation_file_pattern=/data/tf_records/v5-full_merge/validation* \
    --imagenet_checkpoint=/data/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    --logdir=/data/train/v5-full_merge \
    --max_iters=15000 \
    --initial_lr=0.045 \
    --batch_size=50 \
    --num_classes=9
```

### Export Model
```bash
python3 custom_delf/export_local_model.py \
  --ckpt_path=/data/train/v5-full_merge/delf_weights \
  --export_path=/data/models/v5-full_merge
```

## Descubrimiento de estructuras visuales
1. [Extracción de características](#Extracción-de-características)
2. [Clusterización](#Clusterización)
3. [Minado de estructuras](#Minado-de-estructuras)
4. [Visualización de estructuras](#Visualización-de-estructuras)

### Extracción de características 

El script encargado de extraer los descriptores es _extractor.py_ para su ejecución es obligatorio la especificación de tres parámetros:
1. Tipo de extractor
2. Ruta Absoluta o Relativa de la carpeta de imágenes
3. Ruta Absoluta o Relativa y nombre del archivo a generar

La siguiente ejecución creará un archivo cvs con la información de los puntos de interés y un archivo txt con los valores de los descriptores.
```bash
extractor.py SIFT /images_dataset /test/images_descriptors
```

### Clusterización
El proceso de minado requiere de un vocabulario, utilizando los descriptores generados del paso anterior se realiza un clusterizado con la finalidad de estandarizar nuestro vocabulario, al final de este proceso se obtendrá un nuevo archivo que contendrá los un índice que representa el cluster al que fue asociado cada descriptor. 

El script encargado de extraer los descriptores es cluster.py_ para su ejecución es obligatorio la especificación de tres parámetros:
1. Ruta Absoluta o Relativa del archivo generado por el paso anterior
2. Ruta Absoluta o Relativa y nombre del archivo a generar
3. Número de cluster (tamaño de vocabulario final)

La siguiente ejecución creará un archivo Pickle con el nombre _images_clusters_ utilizando 2000 clusters
```bash
cluster.py /test/images_descriptors /test/images_clusters 2000
```
### Minado de estructuras 
Utilizando el archivo generado de la clusterización y el generado de la extracción se crea el doumento de entrada para SHM.

1. Magnitud asociada a los índices de la imagen (Tamaño o frecuencia)
2. Ruta Absoluta o Relativa del archivo generado por el extractor
3. Ruta Absoluta o Relativa del archivo generado por el cluster
4. Ruta Absoluta o Relativa del documento a generar

La siguiente ejecución creará un documento con el nombre _clusters_per_images_ midiendo el tamaño con el que aparecen dentro de la imagen
```bash
SMHdocument.py SIZE /test/images_descriptors /test/images_clusters  /test/clusters_per_images
```
EJECUTAR SMH...

### Visualización de estructuras 

