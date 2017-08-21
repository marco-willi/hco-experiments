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
from tools.subjects import SubjectSet, Subject
import pickle
from datetime import datetime


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

# link_cl = 'https://zooniverse.slack.com/files/adam/F5YQGD5ML/elephant-expedition-classifications.csv.zip'
# link_sub = 'https://zooniverse.slack.com/files/adam/F5Y3FHA13/elephant-expedition-subjects.csv.zip'

#########################
# Get Data & Save
# manual step
#########################

# cfg_path['db']
#
# # save classifications
# create_path(cfg_path['db'])
# path_to_file = get_url(link_cl, cfg_path['db'] + 'classifications.zip')
#
#
# # save subject data
# create_path(cfg_path['db'] + 'subjects')
# path_to_file = get_url(link_cl, cfg_path['db'] + 'subjects.zip')


#########################
# Get Data
#########################

subs = read_subject_data(cfg_path['db'] + 'subjects.csv')
subs[list(subs.keys())[0]]

cls = read_classification_data(cfg_path['db'] + 'classifications.csv')
cls.head

###############################
# Analysis
###############################

# workflows
cls['workflow_name'].unique()
cls.groupby(['workflow_name']).size()

# workflow_name
# Angola               553147
# Empty Or Not              3
# Gabon (1)             61848
# Namibia (1)          292281
# SE Asia (1)          240096
# SW Angola           1835325
# South Africa (3)    1303327
# Tajikistan (1)       408101
# test                     39

# workflow versions
cls.groupby(['workflow_version']).size()

# filter on workflow
cls = cls[cls.workflow_name == 'South Africa (3)']

# filter classifications without a choice
cls = cls[cls.annotations.str.contains('choice')]

# filter classifications without most recent workflow_version
work_v = cls.groupby(['workflow_version']).size()
# workflow_version
# 311.3     142783
# 318.4    1160544
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

# print answers with multiple choices
for i in range(0, len(choic)):
    if len(choic[i]) >1:
        print("----------------------------")
        print(i)
        print(choic[i])

# take a look at an individual multiple choice answer
idd = 1159980
cls.iloc[idd]
cls.iloc[idd]['subject_data']
subs[int(list(json.loads(cls.iloc[idd]['subject_data']).keys())[0])]

# retirement reasons
subd = dict()
retirement_reasons = list()
for i in range(0, cls['subject_data'].shape[0]):
    tt = json.loads(cls['subject_data'].iloc[i])
    key = list(tt.keys())[0]
    if tt[key]['retired'] is None:
        subd[key] = 'Not Retired'
    else:
        subd[key] = tt[key]['retired']['retirement_reason']
    retirement_reasons.append(subd[key])

tt = pd.DataFrame(retirement_reasons, columns=['retirement_reason'])
tt.groupby(['retirement_reason']).size()

# retirement_reason
# Not Retired              75601
# classification_count    238450
# consensus                   53
# nothing_here            146492
# other                   270239
# dtype: int64


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
