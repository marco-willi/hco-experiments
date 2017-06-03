# load modules
from tools import panoptes
from tools.imagedir import ImageDir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.config import config


########################
# Get Info
########################

# get classification & subject data
cls = panoptes.get_classifications(panoptes.my_project)
subs = panoptes.get_subject_info(panoptes.my_project)

project_id = config['projects']['panoptes_id']

########################
# Create Annotations
########################

# encode labels to numerics
labels_all = config[project_id]['classes'].split(",")
le = LabelEncoder()
le.fit(labels_all)

# generate labels from annotations (or somewhere else)
labels = dict()
for key, val in subs.items():
    if '#label' not in val['metadata']:
        next
    else:
        labels[key] = int(le.transform([val['metadata']['#label']]))

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
   data_dict[i] = {'y_data': int(le.transform([val['metadata']['#label']])),
                   'class': val['metadata']['#label'],
                   'url': val['url'],
                   'subject_id': key}
   i +=1

########################
# Test / train /
# validation splits
########################


id_train, id_test = train_test_split(list(data_dict.keys()), train_size=0.95,
                                     random_state=int(config[project_id]
                                                      ['random_seed']))

id_test, id_val = train_test_split(id_test, train_size=0.5,
                                   random_state=int(config[project_id]
                                                    ['random_seed']))



def create_image_dir(data_dict, keys):
    """ Generates directory of images with necessary meta-data """
    # prepare necessary structures
    info_dict = dict()
    ids = list()
    paths = list()
    labels = list()

    # loop through all relevant keys and fill data
    for key in keys:
        dat = data_dict[key]
        ids.append(key)
        paths.append(dat['url'])
        labels.append(dat['y_data'])
        info_dict[key] = data_dict[key]

    # create Image directory object
    img_dir = ImageDir(paths=paths, labels=labels,
                       unique_ids=ids, info_dict=info_dict)
    return img_dir


# generate image directories
train_dir = create_image_dir(data_dict, keys=id_train)
test_dir = create_image_dir(data_dict, keys=id_test)
val_dir = create_image_dir(data_dict, keys=id_val)

# get random image
train_dir.getOneImage(train_dir.unique_ids[0])

# generate some figures
n_subjects = len(subs.keys())
subject_ids = subs.keys()





