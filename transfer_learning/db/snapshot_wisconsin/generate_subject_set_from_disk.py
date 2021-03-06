""" Generate Subject-Set from directory structure """
from config.config import cfg_path, cfg_model
from tools.subjects import SubjectSet, Subject, Image
import os

if __name__ == '__main__':
    # parameters
    path_images = cfg_path['images'] + 'all'

    # Load current subject set to get meta-data
    path_old_subject_set = cfg_path['db'] + 'subject_set_original.json'
    subject_set_old = SubjectSet(labels=cfg_model['classes'])
    subject_set_old.load(path_old_subject_set)

    label_directories = os.listdir(path_images)
    single_label_directories = [x for x in label_directories if '_' not in x]

    # create SubjectSet
    subject_set = SubjectSet(labels=single_label_directories)
    for label in single_label_directories:
        for subject_image in os.listdir(path_images + os.path.sep + label):
            image_identifier = subject_image.split("_")[0]
            old_subject = subject_set_old.getSubject(image_identifier)
            meta_data = old_subject.getMetaData()
            subject = Subject(identifier=image_identifier,
                              labels=label,
                              meta_data=meta_data,
                              urls=old_subject.getURLs()[0]
                              )
            image = subject.getImage(image_identifier + "_0")
            image.setPath(path_images + os.path.sep + label + os.path.sep)
            subject_set.addSubject(subject)
    subject_set.printLabelDistribution()
    # save to disk
    subject_set.save(cfg_path['db'] + 'subject_set.json')

    # test
    subject.getFilePaths()
    subject_set_loaded = SubjectSet(labels=single_label_directories)
    subject_set_loaded.load(cfg_path['db'] + 'subject_set.json')
    subject_set_loaded.getSubject(image_identifier).urls
    subject_set_loaded.getSubject(image_identifier).images
    subject_set_loaded.getSubject(image_identifier).images['9201838_0'].path
