# import modules
import configparser
import os
import logging
from datetime import datetime
import sys

path_cfg = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'config.ini')
path_cred = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'credentials.ini')

# read config files
config_credentials = configparser.ConfigParser()
config_credentials.read(path_cred)

config = configparser.ConfigParser()
config.read(path_cfg)

# identify platform and set behavior of creating links or file copies
platform = sys.platform
if platform == 'win32':
    config['general']['link_only'] = '0'
else:
    config['general']['link_only'] = '1'


# function to load path parameters
def path_loader(config, create_project_paths=True):

    # project_id
    project_id = config['projects']['panoptes_id']

    if eval(config['general']['debug']):
        paths = config['paths_debug']
    else:
        paths = config['paths']

    # create project paths
    if create_project_paths:
        for p in paths:
            # create only if main path exists
            if os.path.exists(paths[p]):
                if not os.path.exists(paths[p] + project_id):
                    os.mkdir(paths[p] + project_id)

    # add project id to paths
    for p in paths:
        paths[p] = paths[p] + project_id + "/"

    return paths


cfg_path = path_loader(config)


def _extract_configs(key, value):
    """ Extract configs to dictionary """
    if key in ['classes', 'callbacks']:
        splitted = value.replace("\n", "").split(",")
        return splitted
    elif key in ['image_size_save', 'image_size_model']:
        size = value.split(',')
        size = tuple([int(x) for x in size])
        return size
    else:
        try:
            return eval(value)
        except:
            return value


# function to load parameters used for model training
def model_param_loader(config=config):
    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    cfg = dict()

    # load general configs
    for key, value in config['general'].items():
        cfg[key] = _extract_configs(key, value)

    # load specific configs and override general if required
    for key, value in config[project_id].items():
        cfg[key] = _extract_configs(key, value)

    return cfg


cfg_model = model_param_loader(config)

print("Config Loaded")


##############################
# Logging
##############################

# initialize logging file
ts = datetime.now().strftime('%Y%m%d%H%m')
logging.basicConfig(filename=cfg_path['logs'] + ts + '_run.log',
                    filemode="w",
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')
