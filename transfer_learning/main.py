# load modules
from tools import panoptes
from tools.imagedir import ImageDir, create_image_dir
from sklearn.model_selection import train_test_split

from config.config import config
import importlib
from db.generate_annotations import generate_annotations_from_panoptes
from tools.subjects import SubjectSet, Subject

# some tests
def test(train_dir):
    # get random image
    train_dir.getOneImage(train_dir.unique_ids[0])


def prep_data():

    ########################
    # Get Project Info
    ########################

    # get classification & subject data via Panoptes client
    # cls = panoptes.get_classifications(panoptes.my_project)
    subs = panoptes.get_subject_info(panoptes.my_project)

    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    ########################
    # Create Subject Set
    ########################

    all_classes = config[project_id]['classes'].split(",")
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

    ########################
    # Test / train /
    # validation splits
    ########################

    id_train, id_test = train_test_split(list(subject_set.getAllIDs()),
                                         train_size=0.95,
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

    ########################
    # Get Data
    ########################
    train_set, test_set, val_set = prep_data()

    ########################
    # Call Model
    ########################

    # retrieve current model file
    project_id = config['projects']['panoptes_id']
    model_file = config[project_id]['model_file']

    # import model file
    model = importlib.import_module('models.' + model_file)

    # train model
    model.train(train_set, test_set, val_set)


if __name__ == "__main__":
    main()






