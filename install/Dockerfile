FROM tensorflow/tensorflow:2.2.0-gpu

COPY models models

RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

WORKDIR /models/research/delf/delf/python/training

COPY setup.py /models/research/delf/setup.py
COPY install_delf.sh install_delf.sh
RUN bash install_delf.sh

RUN apt install protobuf-compiler -y
WORKDIR /models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN python -m pip install .

