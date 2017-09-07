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
    config['general']['multi_processing'] = '0'
else:
    config['general']['link_only'] = '1'
    config['general']['multi_processing'] = '0'


# add timestamp to config
config['general']['ts'] = datetime.now().strftime('%Y%m%d%H%m')


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
            # reformat paths according to os
            paths[p] = os.path.normpath(paths[p]) + os.sep
            # create only if main path exists
            if os.path.exists(paths[p]):
                if not os.path.exists(paths[p] + project_id):
                    os.mkdir(paths[p] + project_id)

    # add project id to paths
    for p in paths:
        paths[p] = paths[p] + project_id + os.sep

    return paths


cfg_path = path_loader(config)


def _extract_configs(key, value):
    """ Extract configs to dictionary """
    if key in ['classes', 'callbacks']:
        splitted = value.replace("\n", "").split(",")
        return splitted
    elif key in ['image_size_save', 'image_size_model']:
        if value in ['', None, 'None']:
            return None
        else:
            size = value.split(',')
            size = tuple([int(x) for x in size])
            return size
    elif key in ['load_model'] and value not in ['', None]:
        return os.path.normpath(value)
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

    # load experiment config if available
    if 'experiment_id' in config[project_id].keys():
        for key, value in config[config[project_id]['experiment_id']].items():
            cfg[key] = _extract_configs(key, value)

    return cfg


cfg_model = model_param_loader(config)

print("Config Loaded")


##############################
# Logging
##############################

# timestamp and logging file name
ts = str(config['general']['ts'])
if 'experiment_id' in cfg_model:
    exp_id = cfg_model['experiment_id'] + '_'
else:
    exp_id = ''

# logging handlers
handlers = list()

if cfg_model['logging_to_disk'] == 1:
    # handlers to log stuff to (file and stdout)
    file_handler = logging.FileHandler(
        filename=cfg_path['logs'] + exp_id + ts + '_run.log')
    handlers.append(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
handlers.append(stdout_handler)

# logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(funcName)s - %(levelname)s:' +
                           '%(message)s',
                    handlers=handlers)

# log parameters / config
logging.info("Path Parameters: %s" % cfg_path)
logging.info("Model Parameters: %s" % cfg_model)
