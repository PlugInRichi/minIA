
cd data
mkdir "tf_records"
mkdir "models"
mkdir -p "images/images_gz2"

wget -O images_gz2.zip https://zenodo.org/record/3565489/files/images_gz2.zip?download=1
zip -d images_gz2.zip "__MACOSX"
unzip -q -j images_gz2.zip -d images/images_gz2
rm images_gz2.zip

cd models
curl -Os http://storage.googleapis.com/delf/resnet50_imagenet_weights.tar.gz
tar -xzvf resnet50_imagenet_weights.tar.gz
rm resnet50_imagenet_weights.tar.gz
