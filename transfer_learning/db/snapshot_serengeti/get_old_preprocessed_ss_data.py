# function go get old Season 1-6 SS data
import urllib.request
import csv
from db.snapshot_serengeti.get_ss_urls import get_oroboros_api_data
import time
import numpy as np
import pickle

#############################
# Parameters
#############################

api_path = 'https://api.zooniverse.org/projects/serengeti/subjects/'


############################
# Get Data From DATA DRYAD
############################


url_data = 'http://datadryad.org/bitstream/handle/10255/dryad.86348/\
consensus_data.csv?sequence=1'

output_file = 'D:/Studium_GD/Zooniverse/Data/snapshot_serengeti/annotations/\
consolidated_annotations.csv'
#response = urllib.request.urlretrieve(url_data, output_file)

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



############################
# Get SS urls
############################

ids = list()
for key, value in data_dict.items():
    ids.append(key)

time_s = time.time()
res = get_oroboros_api_data(ids[0:1000])
print("Took %s minutes to extract image urls" % ((time.time() - time_s)/60))

pickle.dump(res, 'ss_urls.pkl')


############################
# Create Subject Set
############################

#all_classes = config[project_id]['classes'].split(",")
#subject_set = SubjectSet(labels=all_classes)
#
#for key, value in subs.items():
#    subject = Subject(identifier=key,
#                      label=value['metadata']['#label'],
#                      meta_data=value['metadata'],
#                      urls=value['url'],
#                      label_num=subject_set.getLabelEncoder().transform(
#                              [value['metadata']['#label']])
#                      )
#    subject_set.addSubject(key, subject)









