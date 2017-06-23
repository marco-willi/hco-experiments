"""
Code to process data dumps from panoptes
"""
from config.config import cfg_model, cfg_path
import urllib.request
import os
import pandas as pd
import json
import time
import numpy as np
from collections import Counter


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
    subject_urls = [json.loads(x)['0'] for x in subs_df['locations']]
    subject_meta = [json.loads(x) for x in subs_df['metadata']]
    # fill dictionary
    subs_dir = dict()
    for i in range(0, len(subject_ids)):
        subs_dir[subject_ids[i]] = {'url': subject_urls[i],
                                    'metadata': subject_meta[i]}
    return subs_dir

def read_classification_data(path_csv):
    # read csv
    cls_df = pd.read_csv(path_csv)
    return cls_df


#########################
# Parameters
#########################

link_cl = 'https://zooniverse.slack.com/files/adam/F5YQGD5ML/elephant-expedition-classifications.csv.zip'
link_sub = 'https://zooniverse.slack.com/files/adam/F5Y3FHA13/elephant-expedition-subjects.csv.zip'

#########################
# Get Data & Save
# manual step
#########################

cfg_path['db']

# save classifications
create_path(cfg_path['db'])
path_to_file = get_url(link_cl, cfg_path['db'] + 'classifications.zip')


# save subject data
create_path(cfg_path['db'] + 'subjects')
path_to_file = get_url(link_cl, cfg_path['db'] + 'subjects.zip')


#########################
# Process Subject Data
#########################

subs = read_subject_data(cfg_path['db'] + 'subjects.csv')

###############################
# Process Classification Data
###############################

cls = read_classification_data(cfg_path['db'] + 'classifications.csv')
cls.head

# filter on workflow: Spot and count rainforest animals
cls = cls[cls.workflow_name == 'Spot and count rainforest animals']

# filter classifications without a choice
cls = cls[cls.annotations.str.contains('choice')]

# filter classifications without most recent workflow_version
work_v = cls.groupby(['workflow_version']).size()
most_recent_wf = work_v.index[-1]
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


# generate plurality algorithm result
subs_res_final = dict()
for k, v in subs_res.items():

    # check retirement reason
    if v['retirement_reason'] in ['Not Retired','classification_count']:
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
        elif label_final[i] in ['nothing_here', 'VEGETATIONNOANIMAL']:
            label_final[i] = 'blank'

    subs_res_final[k] = {'ret_label': label,
                         'plur_label': label_plur,
                         'n_species': n_species_med,
                         'label': label_final}


subs_res_final[list(subs_res_final.keys())[5070]]

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
        if l == 'VEGETATIONNOANIMAL':
            print(str(k) + "  " + str(v['label']))
        if l not in labels_all:
            labels_all[l] = 1
        else:
            labels_all[l] += 1




# workflows
cls['workflow_name'].unique()
cls.groupby(['workflow_name']).size()

# workflow versions
cls.groupby(['workflow_version']).size()

# filter on workflow: Spot and count rainforest animals
cls = cls[cls.workflow_name == 'Spot and count rainforest animals']

# filter classifications without a choice
cls = cls[cls.annotations.str.contains('choice')]

# filter classifications without most recent workflow_version
work_v = cls.groupby(['workflow_version']).size()
most_recent_wf = work_v.index[-1]
cls = cls[cls.workflow_version == most_recent_wf]

# subject id
print("number of subjects %s" % len(cls['subject_ids'].unique()))


# look for multiple choices per classification
choic = list()
for i in range(0, cls['annotations'].shape[0]):
    tt = json.loads(cls['annotations'].iloc[i])
    choices = [x['choice'] for x in tt[0]['value']]
    choic.append(choices)

for i in range(0, len(choic)):
    if len(choic[i]) >1:
        print("----------------------------")
        print(i)
        print(choic[i])

idd = 1347199
cls.iloc[idd]
cls.iloc[idd]['subject_data']
subs[int(list(json.loads(cls.iloc[idd]['subject_data']).keys())[0])]

# retirement reasons
subd = dict()
retirement_reasons = list()
for i in range(0, cls['subject_data'].shape[0]):
     tt = json.loads(cls['subject_data'].iloc[i])
     key = list(tt.keys())[0]
     if tt[key]['retired'] == None:
         subd[key] = 'Not Retired'
     else:
         subd[key] = tt[key]['retired']['retirement_reason']
     retirement_reasons.append(subd[key])

tt = pd.DataFrame(retirement_reasons, columns=['retirement_reason'])
tt.groupby(['retirement_reason']).size()

# check number of entries per subject and user
cls.columns
tt = cls.groupby(['subject_ids', 'user_name']).size()
tt = tt[tt > 1]
#tt = cls[(cls.subject_ids == 10444100) & (cls.user_name == 'cmdctrl')]

for i in range(0, tt.shape[0]):
    tt2 = cls[(cls.subject_ids == tt.index[i][0]) & (cls.user_name == tt.index[i][1])]
    time.sleep(3)
    print("--------------------------------")
    for ii in range(0, tt2.shape[0]):
        print("Classification ID " + str(tt2.iloc[ii, :]['classification_id']))
        print("User name         " + str(tt2.iloc[ii, :]['user_name']))
        print("Annotations       " + str(tt2.iloc[ii, :]['annotations']))
        print("Subject Data      " + str(tt2.iloc[ii, :]['subject_data']))


# check number of entries per subject and user
cls.columns
tt = cls.groupby(['subject_ids', 'user_name', 'classification_id']).size()
tt = tt[tt > 1]



# workflows
cls.iloc[1672081,:]['metadata']

for i in range(500, 550):
    print("%s: ------------------" % i)
    print(cls.iloc[i,:]['annotations'])






