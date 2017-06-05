# load modules
from tools import panoptes
from tools.imagedir import ImageDir, create_image_dir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.config import config
from models.cat_vs_dog_test_disk_only import train

# some tests
def test(train_dir):
    # get random image
    train_dir.getOneImage(train_dir.unique_ids[0])


def prep_data():

    ########################
    # Get Project Info
    ########################

    # get classification & subject data
    cls = panoptes.get_classifications(panoptes.my_project)
    subs = panoptes.get_subject_info(panoptes.my_project)

    # extract project id for further loading project specifc configs
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
    i = 0
    for key, val in subs.items():
        data_dict[i] = {'y_data': int(le.transform([val['metadata']
                                                       ['#label']])),
                        'class': val['metadata']['#label'],
                        'url': val['url'],
                        'subject_id': key}
        i += 1

    ########################
    # Test / train /
    # validation splits
    ########################

    id_train, id_test = train_test_split(list(data_dict.keys()),
                                         train_size=0.95,
                                         random_state=int(config[project_id]
                                                          ['random_seed']))

    id_test, id_val = train_test_split(id_test,
                                       train_size=0.5,
                                       random_state=int(config[project_id]
                                                        ['random_seed']))

    # generate image directories
    train_dir = create_image_dir(data_dict, keys=id_train)
    test_dir = create_image_dir(data_dict, keys=id_test)
    val_dir = create_image_dir(data_dict, keys=id_val)

    return train_dir, test_dir, val_dir


# Main Program
def main():

    ########################
    # Get Data
    ########################
    train_dir, test_dir, val_dir = prep_data()

    ########################
    # Call Model
    ########################
    train(train_dir, test_dir, val_dir)


if __name__ == "__main__":
    main()






