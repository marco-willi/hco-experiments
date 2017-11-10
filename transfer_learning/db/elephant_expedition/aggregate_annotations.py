""" Aggregate Annotations """
from config.config import cfg_path
import pandas as pd
from pandas.io.json import json_normalize
import json
import csv
import numpy as np
from collections import Counter, OrderedDict
from db.data_prep_functions import *
from datetime import datetime

###############################
# Parameters
###############################

classification_csv_path = cfg_path['db'] + 'new/classifications.csv'
subject_csv_path = cfg_path['db'] + 'subjects.csv'
workflow_name_filter = ('Spot and count rainforest animals')
workflow_version_filter = ('30.3')
result_json_path = cfg_path['db'] + 'result.json'
result_csv_path = cfg_path['db'] + 'result.csv'

###############################
# Import Classification Data
###############################
cls_raw = read_classification_data_camcat(classification_csv_path)

# show some random keys
keys = np.random.choice(list(cls_raw.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    cls_raw[k]
print("Number of Raw Classifications: %s" % len(cls_raw.keys()))

# analyse classification data
workflow_names = list()
workflow_version = list()
for k, v in cls_raw.items():
    workflow_names.append(v['workflow_name'])
    workflow_version.append(v['workflow_version'])
df = pd.DataFrame({'worfklow_names': workflow_names,
                   'workflow_version': workflow_version})
df.groupby('worfklow_names').size()
df.groupby('workflow_version').size()

# choose correct workflow name and version
# workflow_name_filter = ('Spot and count rainforest animals')
# workflow_version_filter = ('30.3')

# create dictionary with filtered classifications
cls = OrderedDict()
for k, v in cls_raw.items():
    if (v['workflow_name'] in workflow_name_filter) and \
       (v['workflow_version'] in workflow_version_filter) and \
       ('choice' in v['annotations']):
        cls[k] = v

# show some random keys
keys = np.random.choice(list(cls.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    cls[k]

print("Number of Filtered Classifications: %s" % len(cls.keys()))


###############################
# Process Subject Data
###############################

subs = read_subject_data_camcat(subject_csv_path)
# show some random keys
keys = np.random.choice(list(subs.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs[k]

# subset all subjects that were used in classifications
subs_used_all = set(v['subject_ids'] for v in cls.values())
subs_used = dict()
for k, v in subs.items():
    if k in subs_used_all:
        subs_used[k] = v

# extract location, date and time of subject
for v in subs_used.values():
    location = v['metadata']['Filename'].split('-')[0].strip()
    date_time = v['metadata']['Filename'].split('-')[-1].strip()
    date = date_time.split(' ')[0].strip()
    time_of_day = date_time.split(' ')[1].split('.')[0].strip()
    try:
        datetime_object = datetime.strptime(date + '_' + time_of_day,
                                            '%Y_%m_%d_%H_%M_%S')
    except:
        print(v['metadata']['Filename'])
    v['location'] = location
    v['date'] = datetime_object.strftime("%Y%m%d")
    v['time'] = datetime_object.strftime("%H%M%S")
    v['datetime'] = datetime_object.strftime("%Y%m%d%H%M%S")


###############################
# Process Classification Data
###############################

# loop through all classifications and fill subject dictionary
subs_res = dict()
for k, v in cls.items():
    # get subject id
    s_id = v['subject_ids']
    # retirement
    ret = json.loads(v['subject_data'])
    key = list(ret.keys())[0]
    if ret[key]['retired'] is None:
        ret_res = 'Not Retired'
    else:
        ret_res = ret[key]['retired']['retirement_reason']
    # classifications
    cls_usr = [x['choice'] for x in json.loads(v['annotations']
                                               )[0]['value']]
    # create user in subject key
    if v['subject_ids'] not in subs_res:
        subs_res[s_id] = {'users': OrderedDict(),
                          'retirement_reason': ''}
    # create dictionary for user
    if v['user_name'] not in subs_res[s_id]['users']:
        subs_res[s_id]['users'][v['user_name']] = OrderedDict()
    # add all classifications / species to subject/user combination
    for cl in cls_usr:
        subs_res[s_id]['users'][v['user_name']][cl] = 1
    subs_res[s_id]['retirement_reason'] = ret_res

# show some random keys
keys = np.random.choice(list(subs_res.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_res[k]


# generate plurality algorithm result
subs_res_final = dict()
for k, v in subs_res.items():
    # blanks
    blank_classes = ['nothing_here', 'VEGETATIONNOANIMAL', 'NOTHINGHERE']
    # check retirement reason
    if v['retirement_reason'] in ['Not Retired', 'classification_count', None]:
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
    # calculate plurality label after each classification
    plur_labels = list()
    for uu in range(0, len(users)):
        # ordered dict
        for u in list(users.keys())[0:uu]:
            n_species_user.append(len(users[u]))
            species_all.extend(list(users[u].keys()))
        n_species_med = np.median(n_species_user)
        top_n = Counter(species_all).most_common(int(n_species_med))
        plur_labels.append([x[0] for x in top_n])
    # extract label
    if label == 'unkwnown':
        label_final = plur_labels[-1]
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
                         'plur_label': plur_labels[-1],
                         'plur_label_history': plur_labels,
                         'n_species': n_species_med,
                         'label': label_final,
                         'n_users': len(users.keys()),
                         'pielou': calc_pielu(species_all, blank_classes)}

# show some random keys
keys = np.random.choice(list(subs_res_final.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_res_final[k]

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
    print("Label %s has %s obs" % (k, v))

###############################
# Combine Subject &
# Classification Data
###############################


# prepare final dictionary that contains all relevant data
subs_all_data = dict()
for k, v in subs_res_final.items():
    # remove all multi label stuff
    label = v['label']
    if label[0] is None:
        print(k)
        print(v)
    if label is None:
        continue
    if k not in subs_used.keys():
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

# show some random keys
keys = np.random.choice(list(subs_all_data.keys()), size=10)
for k in keys:
    print("Key: %s" % k)
    subs_all_data[k]


# check all labels
labels_all = dict()
for k, v in subs_all_data.items():
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
for k, v in labels_all.items():
    print("Label %s has %s obs" % (k, v))


###############################
# Export to Disk
###############################

save_res = dict()
for k, v in subs_all_data.items():
    sub_d = {'label': v['label'],
             'urls': v['url'],
             'meta_data': v['meta_data']}
    save_res[k] = sub_d
# write to json file
with open(result_json_path, 'w') as fp:
    json.dump(save_res, fp, indent=0)

# write csv
res_df_norm = json_normalize([{**{'subject_id': k}, **v} for k, v in save_res.items()])
res_df_norm.to_csv(result_csv_path, index=False)


test = dict()
for k in list(save_res.keys())[0:5]:
    test[k] = save_res[k]
res_df_norm = json_normalize([test])

# write csv
with open('sample.csv') as fh:
    rows = csv.reader(fh, delimiter=',')
    header = next(rows)

    # "transpose" the data. `data` is now a tuple of strings
    # containing JSON, one for each row
    idents, dists, data = zip(*rows)

data = [json.loads(row) for row in data]
df = json_normalize(data)
df['ids'] = idents
df['dists'] = dists




res_df_norm = json_normalize(save_res)
res_df = pd.DataFrame.from_dict(save_res)
