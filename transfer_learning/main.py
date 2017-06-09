# load modules
from tools import panoptes
from tools.imagedir import ImageDir, create_image_dir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config.config import config
import importlib
from models.cat_vs_dog_test_disk_only import train
from db.generate_annotations import generate_annotations_from_panoptes

# some tests
def test(train_dir):
    # get random image
    train_dir.getOneImage(train_dir.unique_ids[0])


def prep_data():

    ########################
    # Get Project Info
    ########################

    # get classification & subject data via Panoptes client
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

    # dictionary with subject_id as key,
    data_dict = generate_annotations_from_panoptes(subs, le)

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

    # retrieve current model file
    project_id = config['projects']['panoptes_id']
    model_file = config[project_id]['model_file']

    # import model file
    model = importlib.import_module('models.' + model_file)

    # train model
    model.train(train_dir, test_dir, val_dir)


if __name__ == "__main__":
    main()






