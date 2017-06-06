# git clone https://github.com/marco-willi/hco-experiments.git ~/code/hco-experiments
cp ~/code/credentials.ini ~/code/hco-experiments/transfer_learning/config/credentials.ini
cd ~/code/hco-experiments/transfer_learning/config/

# run submodule
# python3 -m module.submodule

# mount devices (example device is xvdba)
# see specific device info
# sudo file -s /dev/xvdba
# sudo mkfs -t ext4 /dev/xvdba
# sudo mkdir ~/data2
# sudo mount /dev/xvdba ~/data2


# commit docker changes
# sudo docker commit docker_id tensorflow/tensorflow:nightly-devel-gpu-py3