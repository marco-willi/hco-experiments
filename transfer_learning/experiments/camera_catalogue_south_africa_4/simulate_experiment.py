""" Simulate Experiment on Non-Experiment Data to evaluate differences """
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
    'classifications_experiment_20171208.csv')
cls[list(cls.keys())[0]]

# workflow / retirement data per subject
subs_all = OrderedDict()
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
    subs_all[v['subject_ids']] = new_entry
keys = np.random.choice(list(subs_all.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all[str(k)]

# loop through all classifications and fill subject dictionary
subs_res = OrderedDict()
for k, v in cls.items():
    # get subject id
    current_c = v['subject_ids']
    # get workflow_id
    workflow_id = v['workflow_id']
    # get machine prediction
    sub_data = json.loads(v['subject_data'])
    machine_label = sub_data[current_c]['#machine_prediction']
    machine_probability = sub_data[current_c]['#machine_probability']
    # create subject entry if not in dictionary
    if current_c not in subs_res:
        subs_res[current_c] = {workflow_id: {'users': OrderedDict(),
                                             'retirement_reason': '',
                                             },
                               'machine_label': machine_label,
                               'machine_probability': machine_probability}
    elif workflow_id not in subs_res[current_c]:
        subs_res[current_c][workflow_id] = {'users': OrderedDict(),
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
keys = np.random.choice(list(subs_res.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_res[str(k)]


def gen_plur_label(k_sub, v_sub):
    res = dict()
    workflows = ['5000', '5001', '4963']
    for k, v in v_sub.items():
        if k not in workflows:
            continue
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
        label_plur = [x[0].lower() for x in top_n]
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
        # final dict
        res[k] = {
            'ret_label': label,
            'retirement_reason': v['retirement_reason'],
            'plur_label': label_plur,
            'n_species': n_species_med,
            'label': label_final,
            'n_users': len(users.keys()),
            'pielou': calc_pielu(species_all, blank_classes)}
    # consolidate all workflows
    for wf in workflows:
        if wf in res:
            label_final = res[wf]['plur_label']
            retirement_reason = res[wf]['retirement_reason']
    return {'label_plur': label_final,
            'retirement_reason_plur': retirement_reason}


# loop through all subjects and apply experiment logic
exp_res = OrderedDict()
for subject_id, v in subs_res.items():
    # plur label
    plur_label = gen_plur_label(subject_id, v)
    # machine predictions
    machine_label = v['machine_label']
    machine_prob = v['machine_probability']
    # check workflows and loop sequentially
    workflows_all = ['5000', '5001', '4963']
    retire_label = ''
    for workflow in workflows_all:
        if workflow in v and retire_label is '':
            if workflow == '5000':
                annotations = list(v[workflow]['users'].items())
                first_user = annotations[0][1].keys()
                if 'blank' in first_user and machine_label == 'blank':
                    retire_label = 'blank'
                elif machine_label == 'blank':
                    retire_label = 'no_agreement'
                elif 'blank' in first_user:
                    retire_label = 'no_agreement'
                if v[workflow]['retirement_reason'] == 'nothing_here':
                    retire_label_real = 'blank'
            elif workflow == '5001':
                annotations = list(v[workflow]['users'].items())
                first_user = annotations[0][1].keys()
                if 'vehicle' in first_user and machine_label == 'vehicle':
                    retire_label = 'vehicle'
                elif machine_label == 'vehicle':
                    retire_label = 'no_agreement'
                elif 'vehicle' in first_user:
                    retire_label = 'no_agreement'
            elif workflow == '4963':
                annotations = list(v[workflow]['users'].items())
                first_user = set(annotations[0][1].keys())
                if len(annotations) > 1:
                    second_user = set(annotations[1][1].keys())
                    if machine_label.upper() in (first_user & second_user):
                        retire_label = machine_label
                    else:
                        retire_label = 'no_agreement'
                    # print("----------------------------")
                    # print(first_user)
                    # print(second_user)
                    # print(machine_label)
                    # print(retire_label)
                else:
                    retire_label = 'not_retired'
    exp_res[subject_id] = {**{'machine_label': machine_label.lower(),
                              'machine_prob': machine_prob,
                              'retire_label_exp': retire_label.lower()},
                           **plur_label}
keys = np.random.choice(list(exp_res.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    exp_res[str(k)]

# prepare final dictionary that contains all relevant data
subs_all_data = OrderedDict()
for k, v in subs_all.items():
    if k in exp_res:
        subs_all_data[k] = {**exp_res[k],
                            **v}
keys = np.random.choice(list(subs_all_data.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all_data[str(k)]

# convert to pandas df and export to csv
df = pd.DataFrame.from_dict(subs_all_data, orient='index')
df.to_csv(cfg_path['db'] + 'classifications_experiment_20171208_exp_simulation.csv')



# checks
keys = np.random.choice(list(subs_all.keys()), size=10)
for k in keys:
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("Key: %s" % k)
    print("SUBS ALL DATA -----------------------------------")
    subs_all[str(k)]
    print("-------------------------------------------------")
    print("SUBS RES ----------------------------------------")
    subs_res[str(k)]
    print("-------------------------------------------------")
    print("EXP RES -----------------------------------------")
    exp_res[str(k)]


k = '13092778'
print("-------------------------------------------------")
print("-------------------------------------------------")
print("-------------------------------------------------")
print("Key: %s" % k)
print("SUBS ALL DATA -----------------------------------")
subs_all[str(k)]
print("-------------------------------------------------")
print("SUBS RES ----------------------------------------")
subs_res[str(k)]
print("-------------------------------------------------")
print("EXP RES -----------------------------------------")
exp_res[str(k)]



subs_all['13446']
test_id = '13091486'
for k, v in cls.items():
    if v['subject_ids'] == test_id:
        print(v)
        continue
    # vv = json.loads(v['subject_data'])
    # kk = list(vv.keys())[0]
    # if vv[kk]['subject_id'] == test_id:
    #     print(v)
        continue


# FIND SUBJECT ID
test_id = '13132923'
for k, v in cls.items():
    vv = json.loads(v['subject_data'])
    kk = list(vv.keys())[0]
    if vv[kk]['subject_id'] == test_id:
        print("--------------------")
        print(k)
        print(v['subject_ids'])
        print(vv[kk]['subject_id'])
        print(vv[kk])
        continue

# FIND URL
test_url = 'https://s3-eu-west-1.amazonaws.com/pantherabucketleopard1/S17_20000101_20170627/images/S17__Station20__Camera2__CAM42203__2017-06-19__09-46-37.JPG'
for k, v in cls.items():
    vv = json.loads(v['subject_data'])
    kk = list(vv.keys())[0]
    if vv[kk]['link'] == test_url:
        print("--------------------")
        print(v['subject_ids'])
        print(vv[kk]['subject_id'])
        print(vv[kk])
        continue


# Retirement overview
res_all = dict()
for k, v in subs_res.items():
    # workflows
    for w, vv in v.items():
        if w not in ['5000', '5001', '4963']:
            continue
        if w not in res_all:
            res_all[w] = dict()
            res_all[w]['total'] = 1
        ret = vv['retirement_reason']
        if ret not in res_all[w]:
            res_all[w][ret] = 1
        else:
            res_all[w][ret] += 1
            res_all[w]['total'] += 1

res_all = dict()
for k, v in exp_res.items():
    # workflows
    for w, vv in v.items():
        if w not in ['5000', '5001', '4963']:
            continue
        if w not in res_all:
            res_all[w] = dict()
        ret = vv['retirement_reason']
        if ret not in res_all[w]:
            res_all[w][ret] = 1
        else:
            res_all[w][ret] += 1
