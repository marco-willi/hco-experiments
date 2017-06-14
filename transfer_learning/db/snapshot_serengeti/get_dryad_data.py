# Module to get data stored at MSI at Dryad
from urllib import request
import csv
import numpy as np
from config.config import config, cfg_path

def get_dryad_ss_data(retrieve=True):
    ############################
    # Get Data From DATA DRYAD
    ############################

    # load config
    db_path = cfg_path['db']


    url_data = 'http://datadryad.org/bitstream/handle/10255/dryad.86348/consensus_data.csv?sequence=1'

    output_file = db_path + 'consolidated_annotations.csv'
    if retrieve:
        request.urlretrieve(url_data, output_file)

    ############################
    # Get Data From Local Drive
    ############################

    # read file
    file = open(output_file, "r")
    datareader = csv.reader(file)

    data_dict = dict()
    labels = list()

    counter = 0
    for row in datareader:
        counter += 1
        # capture header
        if counter == 1:
            header = row
            continue
        # filter for one species only
        if not row[6] == '1':
            continue
        # fill data
        info_dict = {name: entry for entry, name in zip(row, header)}
        data_dict[row[0]] = {'y_label': row[7],
                             'info': info_dict}
        labels.append(row[7])
    file.close()


    # get all possible labels
    label_counts = np.unique(labels, return_counts=True)

    label_list = list()
    for l, c in zip(label_counts[0], label_counts[1]):
        print("Label: %s, Counts: %s" % (l, c))
        label_list.append(l)

    return data_dict


if __name__ == "__main__":
    dd = get_dryad_ss_data()
