"""
Functions to access data via Panoptes api
- init_panoptes
    connect to panoptes and return project object
- get_classifications
    gets classifications of a project
- get_subject_info
    gets subjects of a project
- download_dic
"""
# import modules
from panoptes_client import Project, Panoptes
from config.config import config, config_credentials
import urllib.request
import pandas as pd
import json
import time


# connect to Panoptes and return project
def init_panoptes(Panoptes=Panoptes, config_credentials=config_credentials):
    # connect to panoptes
    Panoptes.connect(username=config_credentials['Zooniverse']['username'],
                     password=config_credentials['Zooniverse']['password'])

    # get my project
    project_id = int(config['projects']['panoptes_id'])
    my_project = Project.find(id=project_id)

    return my_project


# function to get classification data
def get_classifications(my_project, new=False):
    """ Function to get classifications """
    # create new export
    if new:
        my_project.generate_export('classifications')
    # read csv
    cls = my_project.get_export('classifications')
    cls_df = pd.read_csv(cls.url)
    return cls_df


# function to get subject infos
def get_subject_info(my_project, new=False):
    """ Function to get subject infos (links) """
    # create new export
    if new:
        my_project.generate_export('subjects')
    # read csv
    subs = my_project.get_export('subjects')
    subs_df = pd.read_csv(subs.url)
    # subject ids / urls / metadata
    subject_ids = subs_df['subject_id']
    subject_urls = [json.loads(x)['0'] for x in subs_df['locations']]
    subject_meta = [json.loads(x) for x in subs_df['metadata']]
    # fill dictionary
    subs_dir = dict()
    for i in range(0, len(subject_ids)):
        subs_dir[subject_ids[i]] = {'url': subject_urls[i],
                                    'metadata': subject_meta[i]}
    return subs_dir


# function to download from dictionary
def download_dic(dat, path, n=-1):
    time_start = time.time()
    i = 0
    for sub_id, sub_url in dat.items():
        i += 1
        urllib.request.urlretrieve(sub_url, path + str(sub_id) + ".jpeg")
        if n == i:
            break
    time_end = time.time()
    time_diff = round(time_end - time_start, 2)
    print("Required %s seconds to read %s images" % (time_diff, i))


if __name__ == '__main__':
    my_project = init_panoptes()
    # get classifications and subject links
    cls = get_classifications(my_project)
    subs = get_subject_info(my_project)

    # path to download to
    path_images = config['paths']['path_test_downloads']

    # download images from dictionary
    download_dic(dat=subs, path=path_images, n=10)
