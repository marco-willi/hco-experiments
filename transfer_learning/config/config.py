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

print("Config Loaded")
