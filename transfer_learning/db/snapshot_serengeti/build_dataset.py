# Module to build entire ss dataset for use in normal main function
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
    data = {**dryad_data, **blanks}

    # get image urls from orobouros API
    ids = list(data.keys())
    url_dict = get_oroboros_api_data(ids)

    # save data
    pickle.dump(url_dict, db_path + 'url_dict.pkl')

    # add urls to dict
    for i in ids:
        data[i]['url'] = url_dict[i]

    # create subject_sets for ss_species
    all_classes = config['ss_species']['classes'].split(",")
    subject_set = SubjectSet(labels=all_classes)

    for key, value in data.items():
        # remove blanks
        if value['y_label'] == 'blank':
            continue
        subject = Subject(identifier=key,
                          label=value['y_label'],
                          meta_data=value['info'],
                          urls=value['url'],
                          label_num=subject_set.getLabelEncoder().transform(
                                  [value['y_label']])
                          )
        subject_set.addSubject(key, subject)

    # save data
    pickle.dump(subject_set, db_path + 'subject_set_ss_species.pkl')

    # create subject_sets for ss_blank
    all_classes = config['ss_blank']['classes'].split(",")
    subject_set = SubjectSet(labels=all_classes)

    for key, value in data.items():
        # blanks
        if value['y_label'] != 'blank':
            cl = 'non_blank'
        else:
            cl = 'blank'

        subject = Subject(identifier=key,
                          label=cl,
                          meta_data=value['info'],
                          urls=value['url'],
                          label_num=subject_set.getLabelEncoder().transform(
                                  [cl])
                          )
        subject_set.addSubject(key, subject)

    # save data
    pickle.dump(subject_set, db_path + 'subject_set_ss_blanks.pkl')

if __name__ == '__main__':
    main()

# blanks[list(blanks.keys())[0]]
# dryad_data[list(dryad_data.keys())[0]]
# data[list(data.keys())[0]]

