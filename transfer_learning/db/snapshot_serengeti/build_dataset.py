# Module to build entire ss dataset for use in normal main function
# yields subject sets
from db.snapshot_serengeti.get_dryad_data import get_dryad_ss_data
from db.snapshot_serengeti.get_blanks_from_data_dump import get_blanks
from db.snapshot_serengeti.get_ss_urls import get_oroboros_api_data
from tools.subjects import Subject, SubjectSet
from config.config import config, cfg_path
import pickle

def main():
    # load config
    db_path = cfg_path['db']

    # get species and blank data set
    dryad_data = get_dryad_ss_data(retrieve=False)
    blanks = get_blanks()

    # combine datasets
    # data = {**dryad_data, **blanks}

    data = dryad_data.copy()
    data.update(blanks)

    # get image urls from orobouros API
    ids = list(data.keys())
    url_dict = get_oroboros_api_data(ids)

    # save data
    pickle.dump(url_dict, open(db_path + 'url_dict.pkl', "wb"))

    # url_dict = pickle.load(open(db_path + 'url_dict.pkl', "rb"))

    # add urls to dict
    for i in ids:
        data[i]['url'] = url_dict[i]

    # create subject_sets for ss_species
    all_classes = config['ss']['classes'].replace("\n", "").split(",")
    subject_set = SubjectSet(labels=all_classes)

    for key, value in data.items():
        subject = Subject(identifier=str(key),
                          label=value['y_label'],
                          meta_data=value['info'],
                          urls=value['url'])
        subject_set.addSubject(str(key), subject)

    # save data
    pickle.dump(subject_set, open(db_path + 'subject_set.pkl',
                                  "wb"))


if __name__ == '__main__':
    main()

# blanks[list(blanks.keys())[0]]
# dryad_data[list(dryad_data.keys())[0]]
# data[list(data.keys())[0]]

