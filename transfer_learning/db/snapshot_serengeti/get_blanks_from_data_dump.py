# Function to get blank images from data dump
# 1) Run process_ss_raw_data.py
# 2) Run prep_script.sh
# 3) Run plur_script.sh
import csv
from config.config import config, cfg_path

def get_blanks():
    blanks_file = cfg_path['db'] + 'blanks_data.csv'

    file = open(blanks_file, "r")
    datareader = csv.reader(file)

    data_dict = dict()
    counter = 0
    for row in datareader:
        counter += 1
        if counter == 1:
            continue
        data_dict[row[0]] = {'y_label': row[1],
                             'info': {'season': row[2],
                                      'capture_event_id': row[3]}}

    file.close()
    return data_dict
