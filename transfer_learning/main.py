# load modules
from tools import panoptes
from tools.imagedir import ImageDir
from sklearn.model_selection import train_test_split


########################
# Get Info
########################

# get classification & subject data
cls = panoptes.get_classifications(panoptes.my_project)
subs = panoptes.get_subject_info(panoptes.my_project)


########################
# Create Annotations
########################

# generate labels from annotations (or somewhere else)
labels = dict()
for key, val in subs.items():
    if '#label' not in val['metadata']:
        next
    else:
        labels[key] = val['metadata']['#label']

# get subjects with labels
subs_remove = subs.keys() - labels.keys()

# remove subjects without label
for rem in subs_remove:
    subs.pop(rem, None)


########################
# Data Directory
########################

# create generic dictionary to be used for the modelling part
# contains generic id, y_label, url, subject_id
data_dict = dict()
i=0
for key, val in subs.items():
   data_dict[i] = {'y_data': val['metadata']['#label'],
                   'url': val['url'],
                   'subject_id': key}
   i +=1

########################
# Test / train /
# validation splits
########################

id_train, id_test = train_test_split(list(data_dict.keys()), train_size = 0.8)
id_test, id_val = train_test_split(id_test, train_size = 0.5)

def create_data_dict(data_dict, keys):
    new_dict = dict()
    for key in keys:
        new_dict[key] = data_dict[key]
    return new_dict

train_dict = create_data_dict(data_dict, keys = id_train)
test_dict = create_data_dict(data_dict, keys = id_test)
val_dict = create_data_dict(data_dict, keys = id_val)

# generate image directories
train_dict = ImageDir(train_dict)
test_dict = ImageDir(test_dict)
val_dict = ImageDir(val_dict)



# generate some figures
n_subjects = len(subs.keys())
subject_ids = subs.keys()





