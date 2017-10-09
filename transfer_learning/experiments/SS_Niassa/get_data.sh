# Get data from MSI to AWS instance """

# navigate on MSI
# ssh itasca
ssh lab
module load python/3.4
module load python3/3.5.2_anaconda4.1.1

# send to queue
qsub -q lab job_copy_images.sh

# install tensorflow
# https://www.tensorflow.org/install/install_linux#InstallingVirtualenv
virtualenv --system-site-packages -p python3 tl
source tl/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow
# ImportError: /lib64/libc.so.6: version `GLIBC_2.14' not found

# count files
du * | wc -l

# get size of directory and subdirectories
du -sh *
du -m * | sort -nr | head -n 20

scp -i ~/keys/zv_test_key.pem /home/packerc/shared/albums/Niassa/Niassa_S1/* ubuntu@ec2-204-236-210-147.compute-1.amazonaws.com:~/data_hdd/images/camera_catalogue/


# Transfer files from MSI to AWS
scp -i ~/keys/dummy_key.pem ~/data/Niassa.tar.gz ubuntu@ec2-52-91-165-6.compute-1.amazonaws.com:~/data_hdd/images/niassa
