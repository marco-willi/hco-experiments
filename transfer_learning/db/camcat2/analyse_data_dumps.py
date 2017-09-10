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
from db.data_prep_functions import *


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

subs = read_subject_data_camcat(cfg_path['db'] + 'subjects.csv')
subs[list(subs.keys())[0]]

cls = read_classification_data_camcat(cfg_path['db'] + 'classifications.csv')
cls[list(cls.keys())[0]]


###############################
# Analysis
###############################

# workflows
dd = dict()
for v in cls.values():
    if v['workflow_name'] not in dd:
        dd[v['workflow_name']] = 1
    else:
        dd[v['workflow_name']] += 1
for k, v in dd.items():
    print("Key: %s Value: %s" % ("{:<20}".format(k), v))

# workflow_name
# Key: Vehicle Or Not       Value: 227717
# Key: South Africa (4)     Value: 1718643
# Key: SE Asia (1)          Value: 265923
# Key: Namibia (1)          Value: 292281
# Key: Angola               Value: 553147
# Key: Tajikistan (1)       Value: 408102
# Key: test                 Value: 39
# Key: SW Angola            Value: 1835325
# Key: Gabon (1)            Value: 61848
# Key: Empty Or Not         Value: 361505

dd = dict()
for v in cls.values():
    key = v['workflow_name'] + '_' + v['workflow_id']
    if key not in dd:
        dd[key] = 1
    else:
        dd[key] += 1
for k, v in dd.items():
    print("Key: %s Value: %s" % ("{:<20}".format(k), v))

# Key: South Africa (4)_2647 Value: 1718643
# Key: SE Asia (1)_3206     Value: 265923
# Key: test_1986            Value: 39
# Key: Gabon (1)_2863       Value: 61848
# Key: Empty Or Not_4403    Value: 361505
# Key: Tajikistan (1)_3210  Value: 408102
# Key: Vehicle Or Not_4636  Value: 227717
# Key: Namibia (1)_2731     Value: 292281
# Key: SW Angola_1643       Value: 1835325
# Key: Angola_2672          Value: 553147


# workflow versions
dd = dict()
for v in cls.values():
    key = v['workflow_version']
    if key not in dd:
        dd[key] = 1
    else:
        dd[key] += 1
for k, v in dd.items():
    print("Key: %s Value: %s" % ("{:<20}".format(k), v))

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
# array([2647, 4403, 4636])


# filter classifications without most recent workflow_version
# work_v = cls.groupby(['workflow_version']).size()
# # workflow_version
# # 311.3     142783
# # 318.4    1160544
# most_recent_wf = work_v.index[-1]
# cls = cls[cls.workflow_version == most_recent_wf]

dd = dict()
for v in cls.values():
    key = v['subject_ids']
    if key not in dd:
        dd[key] = 1
    else:
        dd[key] += 1

# subject id
print("number of subjects %s" % len(dd.keys()))
print("number of subjects %s" % len(subs.keys()))

# look for multiple choices per classification
# choic = list()
# for i in range(0, cls['annotations'].shape[0]):
#     tt = json.loads(cls['annotations'].iloc[i])
#     if type(tt[0]['value']) is list:
#         choices = [x['choice'] for x in tt[0]['value']]
#     else:
#         choices = tt[0]['value']
#     choic.append(choices)
#
# # print answers with multiple choices
# for i in range(0, len(choic)):
#     if (type(choic[i]) is list) and (len(choic[i]) >1):
#         print("----------------------------")
#         print(i)
#         print(choic[i])
#
# # take a look at an individual multiple choice answer
# idd = 1159980
# cls.iloc[idd]
# cls.iloc[idd]['subject_data']
# subs[int(list(json.loads(cls.iloc[idd]['subject_data']).keys())[0])]
#
# # get unique choices
# labels_all = dict()
# for ii in choic:
#     lab = list()
#     if type(ii) is not list:
#         lab.append(ii)
#     else:
#         lab = lab + ii
#     for l in lab:
#         if l not in labels_all:
#             labels_all[l] = 1
#         else:
#             labels_all[l] += 1
# for k, v in labels_all.items():
#     print("Label %s has %s obs" % (k, v))

# Label NTHNGHR has 197125 obs
# Label 0 has 33 obs
# Label BUFFALO has 1728 obs
# Label WLDDG has 2584 obs
# Label NOTHINGHERE has 149372 obs
# Label RDBCK has 6519 obs
# Label FIRE has 60 obs
# Label DKRSTNBK has 12888 obs
# Label RNSBL has 1659 obs
# Label LND has 17635 obs
# Label WILDEBEEST has 4618 obs
# Label NSCT has 3402 obs
# Label PORCUPINE has 2281 obs
# Label MNGS has 3346 obs
# Label ELEPHANT has 3564 obs
# Label JACKALBLACKBACKED has 1248 obs
# Label WLDCT has 1276 obs
# Label ELAND has 632 obs
# Label LN has 6861 obs
# Label HYRAX has 19 obs
# Label ROAN has 27 obs
# Label TTR has 154 obs
# Label BTRDFX has 580 obs
# Label GMSBK has 11539 obs
# Label HUMANNOTVEHICLES has 10695 obs
# Label HRTBSTTSSSB has 9148 obs
# Label ZBR has 71622 obs
# Label RDVRK has 2234 obs
# Label WILDCAT has 47 obs
# Label HYAENASPOTTED has 138 obs
# Label JCKL has 16861 obs
# Label AARDWOLF has 615 obs
# Label MPL has 157953 obs
# Label RDNT has 1009 obs
# Label GENET has 387 obs
# Label SMALLASIANMONGOOSE has 1 obs
# Label POLECAT has 35 obs
# Label HONEYBADGER has 183 obs
# Label KLIPSPRINGER has 90 obs
# Label RDWLF has 1717 obs
# Label unknown answer label has 39 obs
# Label SERVAL has 385 obs
# Label KD has 33015 obs
# Label RODENT has 53 obs
# Label VEHICLE has 62569 obs
# Label LPHNT has 41928 obs
# Label JACKALSIDESTRIPED has 138 obs
# Label HNBRWN has 14910 obs
# Label INSECT has 1030 obs
# Label WLDBST has 50065 obs
# Label BUSHPIG has 1236 obs
# Label LEOPARD has 345 obs
# Label DOMESTICDOG has 2 obs
# Label CARACAL has 105 obs
# Label HPPPTMS has 2943 obs
# Label PANGOLIN has 4 obs
# Label DOMESTICCATTLE has 2 obs
# Label AARDVARK has 172 obs
# Label WARTHOG has 24691 obs
# Label PLCT has 91 obs
# Label RHN has 19392 obs
# Label ZEBRA has 3371 obs
# Label ROANSABLE has 3 obs
# Label BSHPG has 5115 obs
# Label KLPSPRNGR has 2788 obs
# Label RABBITHARE has 1427 obs
# Label LION has 870 obs
# Label REEDBUCKRHEBOK has 228 obs
# Label RPTL has 236 obs
# Label GNT has 1814 obs
# Label BIRD has 3633 obs
# Label GRFF has 40669 obs
# Label BUSHBUCK has 877 obs
# Label HARTEBEEST has 251 obs
# Label PRCPN has 8634 obs
# Label NL has 31284 obs
# Label BRD has 23719 obs
# Label OTTER has 18 obs
# Label AFRICANCIVET has 21 obs
# Label GIRAFFE has 3113 obs
# Label Yes has 302486 obs
# Label BSHBCK has 6571 obs
# Label HORSE has 1 obs
# Label BFFL has 13210 obs
# Label MACAQUE has 10 obs
# Label SRVL has 742 obs
# Label No has 286697 obs
# Label VHCL has 293288 obs
# Label BSHBB has 412 obs
# Label HYAENABROWN has 506 obs
# Label 1 has 30 obs
# Label BABOON has 12351 obs
# Label HMNNTVHCLS has 73593 obs
# Label RBBTHR has 13355 obs
# Label MNKBBN has 92176 obs
# Label MONGOOSE has 330 obs
# Label WATERBUCK has 1101 obs
# Label DUIKER has 833 obs
# Label BAT has 14 obs
# Label NYALA has 5560 obs
# Label BT has 409 obs
# Label WRTHG has 63559 obs
# Label CHTH has 1335 obs
# Label WILDBOAR has 1 obs
# Label BATEAREDFOX has 328 obs
# Label TSESSEBE has 92 obs
# Label IMPALA has 20982 obs
# Label HNBDGR has 1978 obs
# Label GEMSBOK has 201 obs
# Label BUSHBABY has 12 obs
# Label BADGER has 1 obs
# Label PNGLN has 82 obs
# Label CRCL has 751 obs
# Label SPRINGBOK has 365 obs
# Label LPRD has 7483 obs
# Label CHEETAH has 216 obs
# Label DOMESTICANIMAL has 269 obs
# Label MONKEY has 4740 obs
# Label WTRBCK has 9105 obs
# Label ARGALIMARCOPOLOSHEEP has 2 obs
# Label FRCNCVT has 3587 obs
# Label HNSPTTD has 7544 obs
# Label SABLE has 19 obs
# Label KUDU has 2830 obs
# Label WILDDOG has 320 obs
# Label GRYSBOK has 80 obs
# Label HIPPOPOTAMUS has 159 obs
# Label RHINO has 2602 obs
# Label WILDPIGEURASIAN has 1 obs
# Label STEENBOK has 242 obs
# Label DMSTCNML has 7121 obs
# Label REPTILE has 64 obs


# retirement reasons
# subd = dict()
# retirement_reasons = list()
# for i in range(0, cls['subject_data'].shape[0]):
#     tt = json.loads(cls['subject_data'].iloc[i])
#     key = list(tt.keys())[0]
#     if tt[key]['retired'] is None:
#         subd[key] = 'Not Retired'
#     else:
#         subd[key] = tt[key]['retired']['retirement_reason']
#     retirement_reasons.append(subd[key])
#
# tt = pd.DataFrame(retirement_reasons, columns=['retirement_reason'])
# tt.groupby(['retirement_reason']).size()

# retirement_reason
# Not Retired             326987
# classification_count    592214
# consensus               214594
# nothing_here            326815
# other                   309439



# check number of entries per subject and user
# cls.columns
# tt = cls.groupby(['subject_ids', 'user_name']).size()
# tt = tt[tt > 1]
# #tt = cls[(cls.subject_ids == 10444100) & (cls.user_name == 'cmdctrl')]
#
# for i in range(0, tt.shape[0]):
#     tt2 = cls[(cls.subject_ids == tt.index[i][0]) & (cls.user_name == tt.index[i][1])]
#     time.sleep(3)
#     print("--------------------------------")
#     for ii in range(0, tt2.shape[0]):
#         print("Classification ID " + str(tt2.iloc[ii, :]['classification_id']))
#         print("User name         " + str(tt2.iloc[ii, :]['user_name']))
#         print("Annotations       " + str(tt2.iloc[ii, :]['annotations']))
#         print("Subject Data      " + str(tt2.iloc[ii, :]['subject_data']))
#
#
# # check number of entries per subject and user
# cls.columns
# tt = cls.groupby(['subject_ids', 'user_name', 'classification_id']).size()
# tt = tt[tt > 1]
#
#
#
# # workflows
# cls.iloc[1672081,:]['metadata']
#
# for i in range(500, 550):
#     print("%s: ------------------" % i)
#     print(cls.iloc[i,:]['annotations'])
