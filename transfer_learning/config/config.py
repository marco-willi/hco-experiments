# import modules
import configparser
import os

path_cfg = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'config.ini')
path_cred = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'credentials.ini')

# read config files
config_credentials = configparser.ConfigParser()
config_credentials.read(path_cred)

config = configparser.ConfigParser()
config.read(path_cfg)


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


# function to load parameters used for model training
def model_param_loader(config=config):
    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    cfg = dict()

    # load all configs
    for key in config[project_id].keys():
        if key in ['classes', 'callbacks']:
            splitted = config[project_id][key].replace("\n",
                                                       "").split(",")
            cfg[key] = splitted
        elif key in ['image_size_save', 'image_size_model']:
            size = config[project_id][key].split(',')
            size = tuple([int(x) for x in size])
            cfg[key] = size
        else:
            try:
                cfg[key] = eval(config[project_id][key])
            except:
                cfg[key] = config[project_id][key]

    return cfg


cfg_model = model_param_loader(config)

print("Config Loaded")
