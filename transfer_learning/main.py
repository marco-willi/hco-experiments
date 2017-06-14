# load modules
from tools import panoptes
from sklearn.model_selection import train_test_split
from config.config import config, cfg_path
import importlib
# from db.generate_annotations import generate_annotations_from_panoptes
from tools.subjects import SubjectSet, Subject
from tools.save_images_on_disk import save_on_disk
import pickle


# Get / Create a subject set
def get_subject_set():

    ########################
    # Get Project Info
    ########################

    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    # retrieve data mode for current data set
    subject_mode = config[project_id]['subject_mode']

    # classes
    all_classes = config[project_id]['classes'].replace('\n', '').split(",")

    ########################
    # Create Subject Set
    ########################

    if subject_mode == 'panoptes':

        my_project = panoptes.init_panoptes()
        # get classification & subject data via Panoptes client
        # cls = panoptes.get_classifications(panoptes.my_project)
        subs = panoptes.get_subject_info(my_project)

        subject_set = SubjectSet(labels=all_classes)

        for key, value in subs.items():
            subject = Subject(identifier=key,
                              label=value['metadata']['#label'],
                              meta_data=value['metadata'],
                              urls=value['url'],
                              label_num=subject_set.getLabelEncoder().transform(
                                      [value['metadata']['#label']])
                              )
            subject_set.addSubject(key, subject)

    if subject_mode == 'disk':
        file = cfg_path['db'] + 'subject_set.pkl'
        subject_set = pickle.load(open(file, 'rb'))

    return subject_set, all_classes


def make_train_test_split(subject_set, project_id, all_classes):

    ########################
    # Test / train /
    # validation splits
    ########################

    ids, labels = subject_set.getAllIDsLabels()

    id_train, id_test = train_test_split(list(ids),
                                         train_size=0.90,
                                         stratify=labels,
                                         random_state=int(config[project_id]
                                                          ['random_seed']))

    id_test, id_val = train_test_split(id_test,
                                       train_size=0.5,
                                       random_state=int(config[project_id]
                                                        ['random_seed']))

    # generate new subject sets
    train_set = SubjectSet(labels=all_classes)
    test_set = SubjectSet(labels=all_classes)
    val_set = SubjectSet(labels=all_classes)

    set_ids = [id_train, id_test, id_val]
    sets = [train_set, test_set, val_set]
    for si, s in zip(set_ids, sets):
        for i in si:
            sub = subject_set.getSubject(i)
            s.addSubject(i, sub)

    return train_set, test_set, val_set


# Main Program
def main():

    # project id
    project_id = config['projects']['panoptes_id']

    ########################
    # Get Data
    ########################

    subject_set, all_classes = get_subject_set()

    ########################
    # Define Test Train Set
    ########################

    train_set, test_set, val_set = make_train_test_split(subject_set,
                                                         project_id,
                                                         all_classes)

    ########################
    # Save Images on Disk
    ########################

    image_storage = config[project_id]['image_storage']

    if image_storage == 'disk':
        save_on_disk(train_set, config, cfg_path, 'train')
        save_on_disk(test_set, config, cfg_path, 'test')
        save_on_disk(val_set, config, cfg_path, 'val')

    ########################
    # Call Model
    ########################

    # retrieve current model file
    model_file = config[project_id]['model_file']

    # import model file
    model = importlib.import_module('models.' + model_file)

    # train model
    model.train(train_set, test_set, val_set)


if __name__ == "__main__":
    main()






