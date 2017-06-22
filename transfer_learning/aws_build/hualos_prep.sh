#!/bin/bash
cd ~/
git clone https://github.com/fchollet/hualos.git
cd ~/hualos

# build docker file
docker build -t hualos-server:latest .
docker run -d -p 5000:5000 hualos-server
