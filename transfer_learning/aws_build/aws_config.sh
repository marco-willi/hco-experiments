sudo rm -r ~/code/hco-experiments
git clone https://github.com/marco-willi/hco-experiments.git ~/code/hco-experiments
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

# connect to aws instance from lucifer
ssh -i ~/keys/zv_test_key.pem ubuntu@ec2-34-207-210-160.compute-1.amazonaws.com

# transfer files from lucifer to aws instance
scp -i ~/keys/zv_test_key.pem ~/data_hdd/db/ss/subject_set.pkl ubuntu@ec2-54-92-164-22.compute-1.amazonaws.com:~


# Nvidia docker
sudo nvidia-docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# normal docker
sudo docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-gpu-py3 bash

# no gpu docker
sudo docker run -it -v ~/:/host tensorflow/tensorflow:nightly-devel-py3 bash
pip install dill requests panoptes_client pillow aiohttp keras

# add swap space
#cd /var/tmp
#dd if=/dev/zero of=swapfile1 bs=10240 count=1048576
#/sbin/mkswap -c -v1 /var/tmp/swapfile1
#sudo /sbin/swapon /var/tmp/swapfile1
