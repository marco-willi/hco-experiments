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

print("Config Loaded")
