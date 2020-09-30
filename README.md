# minIA
_El proyecto tiene como finalidad encontrar características de imágenes astronómicas que permitan contribuir a su clasificación. Haciendo uso de un vocabulario visual se extraen los objetos visuales por medio de Sampled-MinHashing_

## Comenzando 🚀

### En linux

Crear ambiente

```
virtualenv -p python3.6 env
```

Activar el ambiente

```
source env/bin/activate
```

Instalar librerias especificas de opencv

```
pip install -r requirements.txt
```

## Implementación ⌨️

Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur.

### Ejecución 🕹️

El archivo script encargado de extraer los descriptores es _extractor.py_ para su ejecución es obligatorio la especificación de tres parámetros:
1. Tipo de extractor
2. Ruta Absoluta o Relativa de la carpeta de imágenes
3. Ruta Absoluta o Relativa del archivo a crear

La siguiente ejecución creará un archivo Pickle con el nombre _pruebaDescriptor_SIFT_
> extractor.py SIFT C:\imagenes C:\documents\pruebaDescriptor


## Herramientas 🛠️

_Menciona las herramientas que utilizaste_

* [NombreEnlace](https://github.com/PlugInRichi/) - Mi github
