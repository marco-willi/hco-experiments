""" Code to download raw images for classification """
import csv
import pandas as pd
from tools.image_url_loader import ImageUrlLoader
from tools.predictor_external import PredictorExternal
from config.config import cfg_model, cfg_path
import os

# Parameters
# path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\project_data\\camera_catalogue\\'
# file_name = 'manifest_set_28_2017.09.07.csv'
# path_images = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\camera_catalogue\\exp_south_africa_4\\'

path = cfg_path['db']
path_images = cfg_path['images'] + 'exp_south_africa_4' + os.path.sep
file_name = 'manifest_set_28_2017.09.07.csv'


# import manifest
data = pd.read_csv(path + file_name)
data.head

# extract necessary data
urls = list(data.link)
labels = ['unknown' for x in range(0, len(urls))]
fnames = [str(i) + '_' + str(x) + '_' + str(y) for i, x, y in zip(
            range(0, len(urls)),
            list(data.subject_id),
            list(data.image_name))]
assert len(fnames) == len(urls) == len(labels)


# create url loader
img_loader = ImageUrlLoader(parallel=True)

# store images on disk
n_trials = 10
for i in range(0, n_trials):
    try:
        imgs = img_loader.storeOnDisk(
            urls=urls,
            labels=labels,
            fnames=fnames,
            path=path_images,
            target_size=None,
            chunk_size=100, overwrite=False, create_path=True,
            zooniverse_imgproc=False)
    except:
        print("Next Trial %s/%s" % (i, n_trials))


# score images
model_file = cfg_path['save'] + 'cc_species_v2_201708210308.hdf5'
#model_file = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/models/3663/mnist_testing_201708021008_model_04_0.40.hdf5"

# pre_processing = ImageDataGenerator(rescale=1./255)
# pred_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown"
# output_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown/"
# class_list = [str(i) for i in range(0,10)]
#
# predictor = PredictorExternal(
#     path_to_model=model_file,
#     keras_datagen=pre_processing,
#     class_list=class_list)
#
# predictor.predict_path(path=pred_path, output_path=output_path)
