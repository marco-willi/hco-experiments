#!/bin/bash
cd ~/
git clone https://github.com/fchollet/hualos.git
cd ~/hualos

# build docker file
sudo docker build -t hualos-server:latest .
sudo docker run -d -p 8080:9000 hualos-server
