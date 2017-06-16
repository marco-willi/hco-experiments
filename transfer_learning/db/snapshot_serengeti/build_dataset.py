# Module to build entire ss dataset for use in normal main function
# yields subject sets
from db.snapshot_serengeti.get_dryad_data import get_dryad_ss_data
from db.snapshot_serengeti.get_blanks_from_data_dump import get_blanks
from db.snapshot_serengeti.get_ss_urls import get_oroboros_api_data
from tools.subjects import Subject, SubjectSet
from config.config import config, cfg_path
import pickle
import os

def main():
    # load config
    db_path = cfg_path['db']

    # get species and blank data set
    dryad_data = get_dryad_ss_data(retrieve=False)
    blanks = get_blanks()

    # combine datasets
    data = {**dryad_data, **blanks}

    #data = dryad_data.copy()
    #data.update(blanks)

    # get image urls from orobouros API
    ids = list(data.keys())

    fname = db_path + 'url_dict.pkl'

    # check if file exists
    if os.path.exists(fname):
        url_dict = pickle.load(open(fname, "rb"))
        ids_processed = url_dict.keys()

        # remove already processed ids
        ids_to_get = list(set(ids) - set(ids_processed))
    else:
        ids_to_get = ids

    # check if anything is left to load
    if len(ids_to_get) > 0:
        # define chunks of urls to fetch
        cuts = [x for x in range(0, len(ids_to_get), int(1e5))]
        if cuts[-1] < len(ids_to_get):
            cuts.append(len(ids_to_get))

        # convert chunk sizes to integers
        cuts = [int(x) for x in cuts]

        # loop over chunks and save to disk
        for i in range(0, (len(cuts) - 1)):

            idx = [x for x in range(cuts[i], cuts[i+1])]
            ids_chunk = [ids_to_get[z] for z in idx]
            # read from orobouros API
            url_dict = get_oroboros_api_data(ids_chunk)

            # save data
            if os.path.exists(fname):
                url_dict_disc = pickle.load(open(fname, "rb"))
                url_dict = {**url_dict, **url_dict_disc}

            pickle.dump(url_dict, open(fname, "wb"))

        url_dict = pickle.load(open(fname, "rb"))

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

