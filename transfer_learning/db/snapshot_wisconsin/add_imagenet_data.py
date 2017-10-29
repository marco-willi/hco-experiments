""" Add ImageNet data to subjectset """
from tools.subjects import SubjectSet

# train test json
path = cfg_path['db'] + 'train' + '_subject_set_' +\
       cfg_model['experiment_id'] + '.json'

# load set from disk
project_classes = cfg_model['classes']
train_set = SubjectSet(labels=project_classes)
train_set.load(path)

# add images from folder
path_img = "/host/data_hdd/images/imagenet/WOLF"
train_set.add_dir_to_set(path_img)

path_img = "/host/data_hdd/images/imagenet/BEAR"
train_set.add_dir_to_set(path_img)

# save set to disk
train_set.save(path)
