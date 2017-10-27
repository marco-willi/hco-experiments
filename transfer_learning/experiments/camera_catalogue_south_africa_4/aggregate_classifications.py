"""
Code to process data dumps from panoptes
"""
from config.config import cfg_path
import os
import pandas as pd
import json
import numpy as np
from collections import Counter, OrderedDict
import pickle
import random
from db.data_prep_functions import *
from datetime import datetime
import re
import csv


cls =  read_classification_data_camcat(cfg_path['db'] +\
    'classifications_experiment_20171012.csv')
cls[list(cls.keys())[0]]
# dict_keys(['', 'classification_id', 'user_name', 'user_id', 'user_ip',
#  'workflow_id', 'workflow_name', 'workflow_version', 'created_at',
#  'gold_standard', 'expert', 'metadata', 'annotations', 'subject_data',
#  'subject_ids'])

# extract subject data from classification data
subs_all = OrderedDict()
for k, v in cls.items():
    json_extract = json.loads(v['subject_data'])
    key = list(json_extract.keys())[0]
    subs_all[v['subject_ids']] = json_extract[key]

# print sample value
for kk, vv in json_extract[key].items():
    print("Key: %s Value: %s \\n" % (kk, vv))

# randomly show some results
keys = np.random.choice(list(subs_all.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all[str(k)]

# workflow / retirement data per subject
subs_all_workflow = OrderedDict()
for k, v in cls.items():
    json_extract = json.loads(v['subject_data'])
    key = list(json_extract.keys())[0]
    # retirement data
    if ('retired' in json_extract[key]) and \
       (type(json_extract[key]['retired']) is dict):
        retirement_dict = {'ret_' + k: v for k,
                           v in json_extract[key]['retired'].items()}
        del json_extract[key]['retired']
        new_entry = {**json_extract[key], **retirement_dict}
    else:
        new_entry = json_extract[key]
    # workflow
    wf = v['workflow_id']
    # create entry
    if v['subject_ids'] not in subs_all_workflow:
        subs_all_workflow[v['subject_ids']] = dict()
    subs_all_workflow[v['subject_ids']][wf] = new_entry

# randomly show some results
keys = np.random.choice(list(subs_all_workflow.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all_workflow[str(k)]


# loop through all classifications and fill subject dictionary
subs_res = OrderedDict()
for k, v in cls.items():
    # get subject id
    current_c = v['subject_ids']
    # get workflow_id
    workflow_id = v['workflow_id']
    # create subject entry if not in dictionary
    if current_c not in subs_res:
        subs_res[current_c] = {workflow_id: {'users': dict(),
                                             'retirement_reason': ''}}
    elif workflow_id not in subs_res[current_c]:
        subs_res[current_c][workflow_id] = {'users': dict(),
                                            'retirement_reason': ''}
    # current subject workflow
    sub_wf_current = subs_res[current_c][workflow_id]
    # retirement
    ret = json.loads(v['subject_data'])
    key = list(ret.keys())[0]
    if ret[key]['retired'] is None:
        ret_res = 'Not Retired'
    else:
        ret_res = ret[key]['retired']['retirement_reason']
    # classifications
    cl_extract = json.loads(v['annotations'])[0]['value']
    # map answer based on workflow_id
    if workflow_id == '5001':
        mapper = {'Yes': 'novehicle', 'No': 'vehicle'}
        try:
            cl_extract = mapper[cl_extract]
        except:
            pass
    elif workflow_id == '5000':
        mapper = {'Yes': 'notblank', 'No': 'blank'}
        try:
            cl_extract = mapper[cl_extract]
        except:
            pass
    if type(cl_extract) not in (list, str):
        cls_usr = [str(cl_extract)]
    elif 'choice' in cl_extract[0]:
        cls_usr = [x['choice'] for x in cl_extract]
    else:
        cls_usr = [cl_extract]
    # create dictionary for user
    if v['user_name'] not in sub_wf_current['users']:
        sub_wf_current['users'][v['user_name']] = dict()
    # add all classifications / species to subject/user combination
    for cl in cls_usr:
        sub_wf_current['users'][v['user_name']][cl] = 1
    sub_wf_current['retirement_reason'] = ret_res

subs_res[list(subs_res.keys())[0]]

# randomly show some results
keys = np.random.choice(list(subs_res.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_res[str(k)]

# generate plurality algorithm result
subs_res_final = OrderedDict()
for k_sub, v_sub in subs_res.items():
    # loop over all workflows
    for k, v in v_sub.items():
        # blanks
        blank_classes = ['nothing_here', 'VEGETATIONNOANIMAL',
                         'NOTHINGHERE', 'NTHNGHR', 'blank']
        # check retirement reason
        if v['retirement_reason'] in ['Not Retired', 'classification_count',
                                      'consensus', 'other', None]:
            label = 'unkwnown'
        else:
            label = v['retirement_reason']
        # retirement status
        if v['retirement_reason'] not in ['Not Retired', None]:
            retirement_status = 'retired'
        else:
            retirement_status = 'not_retired'
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
        if label == 'unkwnown':
            label_final = label_plur
        else:
            label_final = label
        if type(label_final) is not list:
            label_final = [label_final]
        for i in range(0, len(label_final)):
            if label_final[i] == 'human':
                label_final[i] = 'HUMAN'
            elif label_final[i] in blank_classes:
                label_final[i] = 'blank'
        if k_sub not in subs_res_final:
            subs_res_final[k_sub] = dict()
        # final dict
        subs_res_final[k_sub][k] = {
            'ret_label': label,
            'retirement_reason': v['retirement_reason'],
            'plur_label': label_plur,
            'n_species': n_species_med,
            'label': label_final,
            'n_users': len(users.keys()),
            'pielou': calc_pielu(species_all, blank_classes)}

subs_res_final[list(subs_res_final.keys())[0]]
# randomly show some results
# Key: 13172213
keys = np.random.choice(list(subs_res_final.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_res_final[str(k)]

# prepare final dictionary that contains all relevant data
subs_all_data = OrderedDict()
for k, v in subs_all_workflow.items():
    for kk, vv in v.items():
        if k in subs_res_final:
            if kk in subs_res_final[k]:
                if k not in subs_all_data:
                    subs_all_data[k] = dict()
                subs_all_data[k][kk] = {**subs_res_final[k][kk],
                                        **vv}

# randomly show some results
# Key: 13172213
keys = np.random.choice(list(subs_all_data.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all_data[str(k)]

# save as csv
with open(cfg_path['db'] + 'classifications_experiment_20171012_converted.csv',
          'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                           quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['subject_id', 'workflow', 'attr', 'value'])
    for k, v in subs_all_data.items():
        for kk, vv in v.items():
            for kkk, vvv in vv.items():
                if type(vvv) is list:
                    vvv = vvv[0]
                row = [k, kk, kkk, str(vvv)]
                csvwriter.writerow(row)
