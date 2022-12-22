# minIA
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/tensorflow-2.2-brightgreen)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
![Maintaner](https://img.shields.io/badge/OpenCV_contrib_python-3.4.2.16-blue)

_**minIA** is an unsupervised learning methiodology for discovering patterns in astronomical images (although it can be applied to any other image collection). We use a customized DELF model to extract simple features from images through a bag of words model we represent each image in the collection to mine patterns using the ![Sampled-MinHashing](https://github.com/gibranfp/Sampled-MinHashing) technique._

 ![Screenshot](data/images/structures_examples/r2_l1000,vv_1000_19-1.png)

## Install minIA 🚀
The whole project can be run from a Docker container, instructions for running it correctly can be found [here](install).

The use of **Sampled-MinHashing** requires a separate installation that is not found in the container, for this you have to follow the instructions in [Sampled-MinHashing](https://github.com/gibranfp/Sampled-MinHashing ) repository.

## Execution :joystick:

### Customization of the DELF model for the extraction of astronomical features
1. [Dataset creation](#Dataset-creation)
3. [Reformating dataset](#Reformating-dataset)
4. [Train](#Train)
5. [Export Model](#Export-model)

#### Dataset creation
To configure the creation of the training dataset for the neural model, it is necessary to specify the parameters in a configuration file:
```bash
# data/config/dataset_config.yml
    galaxyZoo2_path:  /data/images/gz2_hart16.csv
    map_images_path:  /data/images/gz2_filename_mapping.csv
    train_dataset_path: /data/images/gz2_train_dataset_5000
    images_dir_path: /data/images/images_gz2
    images_out_dir_path: /data/images/images_gz2
    class_size: 5000
```
Then just run the script
```
python3 createDataSet.py
```
#### Reformating dataset
```
python3 custom_delf/build_galaxy_image_dataset.py \
  --train_clean_csv_path=/data/images/gz2_train_dataset_5000_with_filter.csv \
  --train_directory=/data/images/images_gz2/  \
  --output_directory=/data/tf_records/v5-full_merge \
  --num_shards=64 \
  --validation_split_size=0.2
```
*** Note: to train on any other dataset it is required to have a format equal to 'GZ_dataset.csv' (with the same number of spaces and line breaks, the name of the images is given without format and the extension must be JPG) like:

```
Categoria_Encabezado,Nombre_encabezado
Categoria_ID,nombre_imagen_1 nombre_imagen_2 ...
Categoria_ID,nombre_imagen_2 nombre_imagen_5 ...
```
*** Note: Each image belongs to only one category
*** Note: The names of the headers must match those written in the script

#### Train
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

#### Export Model
```bash
python3 custom_delf/export_local_model.py \
  --ckpt_path=/data/train/v5-full_merge/delf_weights \
  --export_path=/data/models/v5-full_merge
```

### Discovery of visual patterns
1. [Feature extraction](#Feature-extraction)
2. [Bag of words model](#Bag-of-words-model)
3. [Pattern mining](#Pattern-mining)
4. [Explore patterns!](#Explore-patterns)

#### Feature extraction

The script for feature extraction is _extractor.py_ for its execution it is mandatory to specify three parameters:
1. Extractor type (DELF, SIFT, SURF)
2. Absolute or Relative Path of the image folder
3. Absolute or Relative Path and name of the file to generate

The following execution will create a cvs file with the POI information and a txt file with the descriptor values.
```bash
extractor.py SIFT /images_dataset /test/images_descriptors
```

#### Bag of words model
The mining process requires representing each image using the bag of words model. With the descriptors generated in the previous step we build this vocabulary.
For the execution of _cluster.py_ it is mandatory to specify three parameters:
1. Absolute or Relative Path of the file generated by the previous step
2. Absolute or Relative Path and name of the file to generate
3. Number of cluster (final vocabulary size)

The following execution will create new csv file named _images_clusters_ using 2000 clusters
```bash
cluster.py /test/images_descriptors /test/images_visual_vocabulary 2000
```
#### Pattern mining
Using the file generated in the previous step we create the input document for the mining step.
For the execution of _cluster.py_ it is mandatory to specify two parameters:
1. Magnitude associated with the image indices (SIZE or FRECUENCY)
2. Absolute or Relative Path of the file generated in the previous step
3. Absolute or Relative Path of the document to be generated
Optionaly we can use the _drop_outliers_ in order to reduce the size of words

The following execution will create a document with the name _clusters_per_images_ measuring the size with which they appear within the image
```bash
SMHdocument.py SIZE /test/images_visual_vocabulary data/SMH_files/images_BoW.words
```

If we have Sampled-MinHashing installed we can perform the mining like:

```bash
#This create an inverted file index
    smhcmd ifindex  data/SMH_files/images_BoW.words data/SMH_files/images_BoW.ifs
```
```bash
smhcmd discover -r 2 -l 750 data/SMH_files/images_BoW.ifs data/SMH_models/images_structures.model
```
### Explore patterns!
To perform the exploration of visual structures (patterns) we can use [this notebook](src/structure_match.ipynb) to display the images that belong to the same structure
