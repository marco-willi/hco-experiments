# import modules
import configparser

# read config files
config_credentials = configparser.ConfigParser()
config_credentials.read('config/credentials.ini')

config = configparser.ConfigParser()
config.read('config/config.ini')

print("Config Loaded")
