#!/bin/bash
cd ~/
sudo rm -r ~/hualos
git clone https://github.com/marco-willi/hualos.git
cd ~/hualos

# create Dockerfile from hualos_Dockerfile.sh

# build docker file
sudo docker build -t hualos-server:latest .
sudo docker run -d -p 8080:9000 hualos-server
