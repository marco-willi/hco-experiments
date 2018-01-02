""" Change all Image Filenames from name.jpeg to name_0.jpeg """
from config.config import cfg_path, cfg_model
import os

path_images = cfg_path['images'] + 'all'
label_directories = os.listdir(path_images)

for label in label_directories:
    for image in os.listdir(path_images + os.path.sep + label):
        if '_0' not in image:
            new_prefix = image.split(".")[0] + "_0"
            new_name = new_prefix + ".jpeg"
            os.rename(
                path_images + os.path.sep + label + os.path.sep + image,
                path_images + os.path.sep + label + os.path.sep + new_name)
