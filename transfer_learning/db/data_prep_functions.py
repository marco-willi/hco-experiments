import urllib.request
import os
import pandas as pd
import json
import numpy as np
import csv
from collections import OrderedDict

#########################
# Functions
#########################

# retrieve file from url
def get_url(url, fname):
    """ Retrieves file from url and returns path to file """
    path_to_file = urllib.request.urlretrieve(url, fname)
    return path_to_file


def create_path(path, create_path=True):
    if not os.path.exists(path) & create_path:
        os.mkdir(path)
    else:
        NameError("Path not Found")


def read_subject_data(path_csv):
    subs_df = pd.read_csv(path_csv)
    # subject ids / urls / metadata
    subject_ids = subs_df['subject_id']
    subject_urls = [json.loads(x) for x in subs_df['locations']]
    subject_meta = [json.loads(x) for x in subs_df['metadata']]
    # fill dictionary
    subs_dir = dict()
    for i in range(0, len(subject_ids)):
        subs_dir[subject_ids[i]] = {'url': [x for x in subject_urls[i].values()],
                                    'metadata': subject_meta[i]}
    return subs_dir


def read_subject_data1(path_csv):
    subs_df = pd.read_csv(path_csv)
    # subject ids / urls / metadata
    subject_ids = subs_df['subject_id']
    subject_urls = [json.loads(x)['0'] for x in subs_df['locations']]
    subject_meta = [json.loads(x) for x in subs_df['metadata']]
    # fill dictionary
    subs_dir = dict()
    for i in range(0, len(subject_ids)):
        subs_dir[subject_ids[i]] = {'url': subject_urls[i],
                                    'metadata': subject_meta[i]}
    return subs_dir


def read_subject_data2(path_csv):
    subs_df = pd.read_csv(path_csv)
    # subject ids / urls / metadata
    subject_ids = subs_df['subject_id']
    subject_urls = [json.loads(x)['0'] for x in subs_df['locations']]
    subject_meta = [json.loads(x) for x in subs_df['metadata']]
    subject_workflow_ids = subs_df['workflow_id']
    # fill dictionary
    subs_dir = dict()
    for i in range(0, len(subject_ids)):
        subs_dir[subject_ids[i]] = {'url': subject_urls[i],
                                    'metadata': subject_meta[i],
                                    'workflow_id': subject_workflow_ids[i]}
    return subs_dir


def read_subject_data_camcat(path_csv):
    subs_df_dict = OrderedDict()
    counter = 0
    with open(path_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            else:
                subject_id = row[0]
                workflow_id = row[2]
                subject_urls = json.loads(row[5])['0']
                subject_meta = json.loads(row[4])
                subs_df_dict[subject_id] = {'url': subject_urls,
                                            'metadata': subject_meta,
                                            'workflow_id': workflow_id}
                counter += 1
            if (counter % 100000) == 0:
                print("Processed %s" % counter)
    return subs_df_dict


def read_classification_data(path_csv):
    # read csv
    cls_df = pd.read_csv(path_csv)
    return cls_df


def read_classification_data_camcat(path_csv):
    cls_dict = OrderedDict()
    counter = 0
    with open(path_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if counter == 0:
                header = row
                counter += 1
                continue
            else:
                row_dict = {k: v for k, v in zip(header, row)}
                cls_dict[counter] = row_dict
                counter += 1
            if (counter % 100000) == 0:
                print("Processed %s" % counter)
    return cls_dict

def calc_pielu(votes, blank_classes):
    """ calculate pielous evenness """

    # result
    res = dict()

    # remove blanks
    votes = [x for x in votes if x not in blank_classes]

    # total number of votes
    n = len(votes)

    if n == 0:
        return 0

    # calculate proportions
    for v in votes:
        if v not in res:
            res[v] = 1.0
        else:
            res[v] += 1

    # n different species
    n_species = len(res.keys())

    if n_species == 1:
        return 0

    # calculate proportions
    for r, v in res.items():
        res[r] = v / n

    # calculate pielou
    pp = 0
    for r, v in res.items():
        pp += (v * np.log(v))

    return -(pp) / np.log(n_species)
