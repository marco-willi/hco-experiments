#!/bin/bash

# install cudnn
tar -zxf cudnn-8.0-linux-x64-v5.1.solitairetheme8
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/
sudo apt-get install libcupti-dev

# Install nVidia docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

# Prepare some directories in home to be attached to docker
cd ~/
mkdir code
mkdir data

# get tensorflow (only if you want to build it yourself)
# git clone https://github.com/tensorflow/tensorflow.git ~/code/tensorflow

# launch tensorflow-gpu docker
sudo nvidia-docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# launch locally build tensorflow-gpu docker
# sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# install python packages (missing in official tensorflow docker)
pip install dill requests panoptes_client pillow aiohttp keras scikit-image

# save changes
# sudo docker commit docker_id tensorflow/tensorflow:nightly-devel-gpu-py3
