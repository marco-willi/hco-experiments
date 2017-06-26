"""
Code to process data dumps from panoptes
"""
from config.config import cfg_path
import urllib.request
import os
import pandas as pd
import json
import numpy as np
from collections import Counter
from tools.subjects import SubjectSet, Subject
import pickle
import random
from db.data_prep_functions import *




#########################
# Process Subject Data
#########################

subs = read_subject_data(cfg_path['db'] + 'subjects.csv')

cls = read_classification_data(cfg_path['db'] + 'classifications.csv')
cls.head

###############################
# Process Classification Data
###############################

# filter on workflow: Spot and count rainforest animals
cls = cls[cls.workflow_name == 'Initial Workflow']

# filter classifications without a choice
cls[cls.annotations.str.contains('choice')].shape
cls = cls[cls.annotations.str.contains('choice')]

# filter classifications without most recent workflow_version
work_v = cls.groupby(['workflow_version']).size()
most_recent_wf = work_v.index[-1]
cls[cls.workflow_version == most_recent_wf].shape
cls = cls[cls.workflow_version == most_recent_wf]

# loop through all classifications and fill subject dictionary
subs_res = dict()
for i in range(0, cls.shape[0]):

    # get subject id
    current_c = cls.iloc[i]

    # retirement
    ret = json.loads(current_c['subject_data'])
    key = list(ret.keys())[0]
    if ret[key]['retired'] is None:
        ret_res = 'Not Retired'
    else:
        ret_res = ret[key]['retired']['retirement_reason']

    # classifications
    cls_usr = [x['choice'] for x in json.loads(current_c['annotations']
                                               )[0]['value']]

    # create user in subject key
    if current_c['subject_ids'] not in subs_res:
        subs_res[current_c['subject_ids']] = {'users': dict(),
                                              'retirement_reason': ''}

    # create dictionary for user
    if current_c['user_name'] not in subs_res[current_c['subject_ids']]['users']:
        subs_res[current_c['subject_ids']]['users'][current_c['user_name']] = dict()

    # add all classifications / species to subject/user combination
    for cl in cls_usr:
        subs_res[current_c['subject_ids']]['users'][current_c['user_name']][cl] = 1

    subs_res[current_c['subject_ids']]['retirement_reason'] = ret_res

subs_res[list(subs_res.keys())[0]]

# generate plurality algorithm result
subs_res_final = dict()
for k, v in subs_res.items():

    # blanks
    blank_classes = ['nothing_here', 'VEGETATIONNOANIMAL', 'NOTHINGHERE']

    # check retirement reason
    if v['retirement_reason'] in ['Not Retired','classification_count', None]:
        label = 'not_retired'
    else:
        label = v['retirement_reason']

    users = v['users']
    # calculate different species
    n_species_user = list()
    species_all = list()
    for u in users.keys():
        n_species_user.append(len(users[u]))
        species_all.extend(list(users[u].keys()))

    n_species_med = np.median(n_species_user)

    # rank species
    top_n = Counter(species_all).most_common(int(n_species_med))

    # extract label
    label_plur = [x[0] for x in top_n]

    if not label == 'not_retired':
        label_final = label
    else:
        label_final = label_plur

    if type(label_final) is not list:
        label_final = [label_final]
    for i in range(0, len(label_final)):
        if label_final[i] == 'human':
            label_final[i] = 'HUMAN'
        elif label_final[i] in blank_classes:
            label_final[i] = 'blank'



    subs_res_final[k] = {'ret_label': label,
                         'plur_label': label_plur,
                         'n_species': n_species_med,
                         'label': label_final,
                         'n_users': len(users.keys()),
                         'pielou': calc_pielu(species_all, blank_classes)}

subs_res_final[list(subs_res_final.keys())[0]]

# check all labels
labels_all = dict()
for k, v in subs_res_final.items():
    if type(v['label']) is not list:
        lab = list()
        lab.append(v['label'])
    else:
        #print(v['label'])
        lab = v['label']

    for l in lab:
        if l not in labels_all:
            labels_all[l] = 1
        else:
            labels_all[l] += 1

# prepare final dictionary that contains all relevant data
subs_all_data = dict()
for k, v in subs_res_final.items():
    # remove all multi label stuff
    label = v['label']
    if label[0] is None:
        print(k)
        print(v)
    if (len(label) == 1) & (label is not None):
        label = label[0]
    else:
        continue

    if k in subs.keys():
        url = subs[k]['url']
    else:
        continue

    subs_all_data[k] = {'label': label,
                        'url': url,
                        'meta_data': v}

subs_all_data[list(subs_all_data.keys())[0]]

# check all labels
labels_all = dict()
for k, v in subs_all_data.items():
    if type(v['label']) is not list:
        lab = list()
        lab.append(v['label'])
    else:
        print(v['label'])
        lab = v['label']
    for l in lab:
        if l not in labels_all:
            labels_all[l] = 1
        else:
            labels_all[l] += 1

labels_all

# write labels to disk
file = open(cfg_path['db'] + 'classes.txt', "w")
file2 = open(cfg_path['db'] + 'classes_numbers.txt', "w")
for k, v in labels_all.items():
    file.write(str(k) + ',')
    file2.write(str(k) + ',' + str(v) + "\n")

file.close()
file2.close()


# create SubjectSet
subject_set = SubjectSet(labels=list(labels_all.keys()))

for key, value in subs_all_data.items():

    subject = Subject(identifier=key,
                      label=value['label'],
                      meta_data=value['meta_data'],
                      urls=value['url']
                      )
    subject_set.addSubject(str(key), subject)

# save to disk
pickle.dump(subject_set, open(cfg_path['db'] + 'subject_set.pkl',
                              "wb"), protocol=4)

# checks
urls, labels, ids = subject_set.getAllURLsLabelsIDs()

for i in range(0, 50):
    ii = random.randint(0, len(urls))
    print("%s is a %s on: %s" % (ids[ii], labels[ii], urls[ii]))

