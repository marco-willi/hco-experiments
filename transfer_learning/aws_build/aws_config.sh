sudo rm -r ~/code/hco-experiments
git clone https://github.com/marco-willi/hco-experiments.git ~/code/hco-experiments
cp ~/code/credentials.ini ~/code/hco-experiments/transfer_learning/config/credentials.ini
cd ~/code/hco-experiments/transfer_learning/config/


# get specific branch
sudo rm -r ~/code/hco-experiments
git clone -b subjec_set_enhancement https://github.com/marco-willi/hco-experiments.git ~/code/hco-experiments
cp ~/code/credentials.ini ~/code/hco-experiments/transfer_learning/config/credentials.ini
cd ~/code/hco-experiments/transfer_learning/config/


# run submodule
# python3 -m module.submodule

# mount devices (example device is xvdf)
# see specific device info
#sudo file -s /dev/xvdf
#sudo mkfs -t ext4 /dev/xvdf
mkdir ~/data_hdd
sudo mount /dev/xvdf ~/data_hdd

# sudo chmod 775 data_hdd/

# commit docker changes
# sudo docker commit docker_id tensorflow/tensorflow:nightly-devel-gpu-py3
# sudo docker commit docker_id root/tensorflow:latest-devel-gpu-py3


# Nvidia docker
sudo nvidia-docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# Local Nvidia docker
sudo nvidia-docker run -it -v ~/:/host root/tensorflow:latest-devel-gpu-py3 bash

# normal docker
sudo docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# no gpu docker
sudo docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-py3 bash
pip install dill requests panoptes_client pillow aiohttp keras h5py

# connect to aws instance from lucifer
ssh -i ~/keys/zv_test_key.pem ubuntu@ec2-34-207-210-160.compute-1.amazonaws.com

# transfer files from lucifer to aws instance
scp -i ~/keys/zv_test_key.pem ~/data_hdd/db/elephant_expedition/classifications.csv ubuntu@ec2-54-88-211-7.compute-1.amazonaws.com:~/data_hdd/db//elephant_expedition/

# transfer files from new lucifer to aws instance
scp -i ~/keys/zv_test_key.pem /data/lucifer1.2/users/will5448/data_hdd/images/camera_catalogue/* ubuntu@ec2-204-236-210-147.compute-1.amazonaws.com:~/data_hdd/images/camera_catalogue/

# transfer files from aws to aws instance
sudo chmod -R 777 snapshot_wisconsin/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/db/camera_catalogue/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/db/camera_catalogue/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/db/elephant_expedition/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/db/elephant_expedition/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/db/snapshot_wisconsin/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/db/snapshot_wisconsin/

scp -i ~/keys/zv_test_key.pem ~/data_hdd/logs/camera_catalogue/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/logs/camera_catalogue/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/logs/elephant_expedition/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/logs/elephant_expedition/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/logs/snapshot_wisconsin/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/logs/snapshot_wisconsin/

scp -i ~/keys/zv_test_key.pem ~/data_hdd/save/camera_catalogue/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/save/camera_catalogue/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/save/elephant_expedition/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/save/elephant_expedition/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/save/snapshot_wisconsin/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/save/snapshot_wisconsin/

scp -i ~/keys/zv_test_key.pem ~/data_hdd/models/camera_catalogue/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/models/camera_catalogue/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/models/elephant_expedition/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/models/elephant_expedition/
scp -i ~/keys/zv_test_key.pem ~/data_hdd/models/snapshot_wisconsin/* ubuntu@ec2-52-90-191-127.compute-1.amazonaws.com:~/data_hdd/models/snapshot_wisconsin/

# add swap space
#cd /var/tmp
#dd if=/dev/zero of=swapfile1 bs=10240 count=1048576
#/sbin/mkswap -c -v1 /var/tmp/swapfile1
#sudo /sbin/swapon /var/tmp/swapfile1

# Monitor GPU utilization
# nvidia-smi -l 1

# dettach from docker without closing it
# ctrl+ p ctrl + q


# Lucifer - Create new virtualenv
./bin/python3.6/bin/virtualenv -p /home/will5448/bin/python3.6/bin/python3.6 /home/will5448/virtualenvs/tf
./bin/python3.6/bin/virtualenv -p python3 /home/will5448/virtualenvs/tf2
./bin/python3.6/bin/virtualenv -p python3 /home/will5448/virtualenvs/tl2
