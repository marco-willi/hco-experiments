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
from datetime import datetime
import re


##########################
## Parameters
##########################
#
#link_cl = 'https://zooniverse.slack.com/files/adam/F5YQGD5ML/elephant-expedition-classifications.csv.zip'
#link_sub = 'https://zooniverse.slack.com/files/adam/F5Y3FHA13/elephant-expedition-subjects.csv.zip'
#
##########################
## Get Data & Save
## manual step
##########################
#
#cfg_path['db']
#
## save classifications
#create_path(cfg_path['db'])
#path_to_file = get_url(link_cl, cfg_path['db'] + 'classifications.zip')
#
#
## save subject data
#create_path(cfg_path['db'] + 'subjects')
#path_to_file = get_url(link_cl, cfg_path['db'] + 'subjects.zip')


#########################
# Import Subject Data
#########################

subs = read_subject_data_camcat(cfg_path['db'] + 'subjects.csv')
subs[list(subs.keys())[0]]

###############################
# Import Classification Data
###############################

cls = read_classification_data_camcat(cfg_path['db'] + 'classifications.csv')
cls[list(cls.keys())[0]]

for k in list(cls.keys()):
    v = cls[k]
    if v['workflow_name'] not in ['South Africa (4)', 'Vehicle Or Not',
                                  'Empty Or Not']:
        del cls[k]
    elif ('choice' not in v['annotations']) and\
         ('value' not in v['annotations']):
        del cls[k]

#[{"task":"T0","task_label":"Are there any animals, people, or vehicles in this photo?","value":"Yes"}]

# get all workflow ids
dd = dict()
for v in cls.values():
    key = v['workflow_id']
    if key not in dd:
        dd[key] = 1
    else:
        dd[key] += 1
for k, v in dd.items():
    print("Key: %s Value: %s" % ("{:<20}".format(k), v))

work_ids = list(dd.keys())


class_mapper = {
    'ARGALIMARCOPOLOSHEEP': 'ARGALIMARCOPOLOSHEEP',
    'GRFF': 'giraffe',
    'GIRAFFE': 'giraffe',
    'BSHBCK': 'bushbuck',
    'BUSHBUCK': 'bushbuck',
    'BSHBB': 'bushbaby',
    'BUSHBABY': 'bushbaby',
    'PRCPN': 'porcupine',
    'PORCUPINE': 'porcupine',
    'HRTBSTTSSSB': 'hartebeest',
    'HARTEBEEST': 'hartebeest',
    'CHTH': 'cheetah',
    'CHEETAH': 'cheetah',
    'HMNNTVHCLS': 'human',
    'HUMANNOTVEHICLES': 'human',
    'RDBCK': 'reedbuck',
    'REEDBUCKRHEBOK': 'reedbuck',
    'RDVRK': 'aardvark',
    'AARDVARK': 'aardvark',
    'MNKBBN': 'monkeybaboon',
    'BABOON': 'baboon',
    'MONKEY': 'monkey',
    'FRCNCVT': 'africancivet',
    'AFRICANCIVET': 'africancivet',
    'BRD': 'bird',
    'BIRD': 'bird',
    'BSHPG': 'bushpig',
    'BUSHPIG': 'bushpig',
    'HPPPTMS': 'hippopotamus',
    'HIPPOPOTAMUS': 'hippopotamus',
    'DKRSTNBK': 'duikersteenbok',
    'DUIKER': 'duikersteenbok',
    'STEENBOK': 'steenbok',
    'WLDCT': 'wildcat',
    'WILDCAT': 'wildcat',
    'LPHNT': 'elephant',
    'ELEPHANT': 'elephant',
    'KD': 'kudu',
    'KUDU': 'kudu',
    'LPRD': 'leopard',
    'LEOPARD': 'leopard',
    'GEMSBOK': 'GEMSBOK',
    'JCKL': 'jackal',
    'JACKALBLACKBACKED': 'jackalblackbacked',
    'JACKALSIDESTRIPED': 'jackalsidestriped',
    'RBBTHR': 'rabbithare',
    'RABBITHARE': 'rabbithare',
    'HNBRWN': 'hyaenabrown',
    'HYAENABROWN': 'hyaenabrown',
    'DMSTCNML': 'domesticanimal',
    'DOMESTICANIMAL': 'domesticanimal',
    'LND': 'eland',
    'ELAND': 'eland',
    'TTR': 'otter',
    'OTTER': 'otter',
    'HORSE': 'HORSE',
    'WTRBCK': 'waterbuck',
    'WATERBUCK': 'waterbuck',
    'SRVL': 'serval',
    'SERVAL': 'serval',
    'RDWLF': 'aardwolf',
    'AARDWOLF': 'aardwolf',
    'HYRAX': 'hyrax',
    'GNT': 'genet',
    'GENET': 'genet',
    'GRYSBOK': 'GRYSBOK',
    'NSCT': 'insect',
    'INSECT': 'insect',
    'HNSPTTD': 'hyaenaspotted',
    'HYAENASPOTTED': 'hyaenaspotted',
    'GMSBK': 'gemsbock',
    'RHN': 'rhino',
    'RHINO': 'rhino',
    'PLCT': 'polecat',
    'POLECAT': 'polecat',
    'WLDBST': 'wildebeest',
    'WILDEBEEST': 'wildebeest',
    'HNBDGR': 'honeyBadger',
    'HONEYBADGER': 'honeyBadger',
    'BADGER': 'BADGER',
    'WRTHG': 'warthog',
    'WARTHOG': 'warthog',
    'LN': 'lion',
    'LION': 'lion',
    'KLPSPRNGR': 'klipspringer',
    'KLIPSPRINGER': 'klipspringer',
    'BT': 'bat',
    'BAT': 'bat',
    'MACAQUE': 'MACAQUE',
    'PNGLN': 'pangolin',
    'PANGOLIN': 'pangolin',
    'ZBR': 'zebra',
    'ZEBRA': 'zebra',
    'RPTL': 'reptile',
    'REPTILE': 'reptile',
    'VEHICLE': 'vehicle',
    'NL': 'nyala',
    'NYALA': 'nyala',
    'RDNT': 'rodent',
    'RODENT': 'rodent',
    'MPL': 'impala',
    'IMPALA': 'impala',
    'RNSBL': 'roansable',
    'ROANSABLE': 'roansable',
    'ROAN': 'roan',
    'SABLE': 'sable',
    'SMALLASIANMONGOOSE': 'SMALLASIANMONGOOSE',
    'SPRINGBOK': 'SPRINGBOK',
    'TSESSEBE': 'TSESSEBE',
    'BFFL': 'buffalo',
    'BUFFALO': 'buffalo',
    'BTRDFX': 'batEaredFox',
    'BATEAREDFOX': 'batEaredFox',
    'MNGS': 'mongoose',
    'MONGOOSE': 'mongoose',
    'WLDDG': 'wilddog',
    'WILDDOG': 'wilddog',
    'WILDPIGEURASIAN': 'WILDPIGEURASIAN',
    'DOMESTICDOG': 'domesticdog',
    'DOMESTICCATTLE': 'domesticcattle',
    'CRCL': 'caracal',
    'CARACAL': 'caracal',
    'VHCL': 'vehicle',
    'FIRE': 'fire',
    'WILDBOAR': 'WILDBOAR',
    'unknown answer': 'unknown'
}

###############################
# Process Subject Data
###############################

# subset all subjects that were used
# sub_ids_in_cls = set(cls['subject_ids'])
subs_used = dict()
for k, v in subs.items():
    if v['workflow_id'] in work_ids:
        subs_used[k] = v

subs_used[list(subs_used.keys())[0]]
# extract location, date and time of subject
# patterns: '2017-03-09_07-29-18-CAM40480.jpg'
#  'Station22__Camera1__2012-05-14__23-14-29(6).JPG'
# '2016-12-03_10_28_20-.jpg'

# patterns in used subjects:
# t1 = '2017-03-09_07-29-18-CAM40480.jpg'
# t2 = '2016-12-03_10_28_20-.jpg'
# t3 = 'C1940694.JPG'
# t4 = '12_CS4_41628_20160608_094310.jpg'

# t5 = 'S2__Station46__Camera1__CAM43291__2017-04-09__07-30-35.JPG'
# t6 = '2017-03-09_07-29-18-CAM40480.jpg'
# extract_loc_date_time(t5)

def extract_loc_date_time(tt):
    # extract date
    date_regexp = re.search(".*([0-9]{4}[_-]+[0-9]{2}[_-]+[0-9]{2}).*", tt)
    if bool(date_regexp):
        date_extracted = date_regexp.groups()[0]
        date = re.sub('[_-]', '', date_extracted)
    else:
        date = "unkwnown"
    # extract time of day
    tod_regexp = re.search(".*(\D[0-9]{2}[_-]+[0-9]{2}[_-]+[0-9]{2})\D.*", tt)
    if bool(tod_regexp):
        tod_extracted = tod_regexp.groups()[0]
        time_of_day = re.sub('[_-]', '', tod_extracted)
    else:
        time_of_day = "000000"
    # extract location
    loc_regexp = re.search("(CAM[0-9]+)", tt, re.IGNORECASE)
    # loc_regexp = re.search("(?:^.*[_-]|^)(\w+)\.jpg$", tt, re.IGNORECASE)
    if bool(loc_regexp):
        location = loc_regexp.groups()[0]
    else:
        location = "unknown"
    date_time = date + time_of_day
    return location, date, time_of_day, date_time


for v in subs_used.values():
    if 'image_name' in v['metadata']:
        try:
            location, date, time_of_day, date_time = \
                extract_loc_date_time(v['metadata']['image_name'])
        except:
            print(v['metadata']['image_name'])
    elif 'Filename' in v['metadata']:
        try:
            location, date, time_of_day, date_time = \
                extract_loc_date_time(v['metadata']['Filename'])
        except:
            print(v['metadata']['Filename'])
    v['location'] = location
    v['date'] = date
    v['time'] = time_of_day
    v['datetime'] = date_time

subs_used[list(subs.keys())[0]]
subs_used[list(subs.keys())[1000]]
#subs_used[list(subs_used.keys())[0]]

###############################
# Process Classification Data
###############################

# loop through all classifications and fill subject dictionary
subs_res = dict()
for k, v in cls.items():
    # get subject id
    current_c = v['subject_ids']
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
    workflow_id = v['workflow_id']
    if workflow_id == '4636':
        mapper = {'Yes': 'novehicle', 'No': 'vehicle'}
        try:
            cl_extract = mapper[cl_extract]
        except:
            pass
    elif workflow_id == '4403':
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
    # map correct classes
    for ii in range(0, len(cls_usr)):
        if cls_usr[ii] in class_mapper.keys():
            cls_usr[ii] = class_mapper[cls_usr[ii]]
    # create user in subject key
    if v['subject_ids'] not in subs_res:
        subs_res[v['subject_ids']] = {'users': dict(),
                                      'retirement_reason': ''}
    # create dictionary for user
    if v['user_name'] not in subs_res[v['subject_ids']]['users']:
        subs_res[v['subject_ids']]['users'][v['user_name']] = dict()
    # add all classifications / species to subject/user combination
    for cl in cls_usr:
        subs_res[v['subject_ids']]['users'][v['user_name']][cl] = 1
    subs_res[v['subject_ids']]['retirement_reason'] = ret_res

subs_res[list(subs_res.keys())[0]]


# generate plurality algorithm result
subs_res_final = dict()
for k, v in subs_res.items():
    # blanks
    blank_classes = ['nothing_here', 'VEGETATIONNOANIMAL',
                     'NOTHINGHERE', 'NTHNGHR', 'blank']
    # check retirement reason
    if v['retirement_reason'] in ['Not Retired', 'classification_count',
                                  'consensus', 'other', None]:
        label = 'unkwnown'
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
        lab = v['label']
    for l in lab:
        if l not in labels_all:
            labels_all[l] = 1
        else:
            labels_all[l] += 1
for k, v in labels_all.items():
    print("Label %s has %s obs" % ("{:<20}".format(k), v))

# prepare final dictionary that contains all relevant data
subs_all_data = dict()
remove_labels = ['1', '0', 'unknown answer label', 'unknown']
for k in subs_used.keys():
    if k in subs_res_final:
        v = subs_res_final[k]
    else:
        v = dict()
        v['label'] = ['unknown']
    # remove all multi label stuff
    label = v['label']
    if label[0] is None:
        print(k)
        print(v)
    # if (len(label) == 1) & (label is not None):
    #     label = label[0]
    # else:
    #     continue
    if type(label) is list:
        label = [x for x in label if x not in remove_labels]
        if len(label) == 0:
            continue
    elif label in remove_labels:
        continue
    # get subject meta data
    current_sub = subs_used[k]
    sub_meta = {'location': current_sub['location'],
                'date': current_sub['date'],
                'time': current_sub['time'],
                'datetime': current_sub['datetime']}
    subs_all_data[k] = {'label': label,
                        'url': current_sub['url'],
                        'meta_data': {**v, **sub_meta}}

subs_all_data[list(subs_all_data.keys())[0]]

# check all labels
labels_all = dict()
for k, v in subs_all_data.items():
    if type(v['label']) is not list:
        lab = list()
        lab.append(v['label'])
    else:
        lab = v['label']
    for l in lab:
        if l not in labels_all:
            labels_all[l] = 1
        else:
            labels_all[l] += 1
for k, v in labels_all.items():
    print("Label %s has %s obs" % (k, v))

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
                      labels=value['label'],
                      meta_data=value['meta_data'],
                      urls=value['url']
                      )
    subject_set.addSubject(subject)
# save to disk
subject_set.save(cfg_path['db'] + 'subject_set.json')

#pickle.dump(subject_set, open(cfg_path['db'] + 'subject_set2.pkl',
#                              "wb"), protocol=4)

# checks
urls, labels, ids = subject_set.getAllURLsLabelsIDs()
for i in range(0, 200):
    ii = random.randint(0, len(urls))
    print("%s is a %s on: %s" % (ids[ii], labels[ii], urls[ii]))
