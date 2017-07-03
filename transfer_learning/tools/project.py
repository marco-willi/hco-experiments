"""
Class to implement a Project
- defines a set of images and their classes to train a model from
- typically this would be a single Zooniverse project but can also be
  a collection of several projects
"""
from tools import panoptes
from tools.subjects import SubjectSet, Subject
import pickle
from config.config import cfg_model as cfg
from config.config import logging


class Project(object):
    """ class to implement a Zooniverse project """
    def __init__(self, name, panoptes_id,
                 classes, cfg_path, config):
        self.name = name
        self.panoptes_id = panoptes_id
        self.subject_set = None
        self.classes = classes
        self.cfg_path = cfg_path
        self.cfg = cfg

    def createSubjectSet(self, mode):
        """ Function to Create a Full Subject Set """

        if mode == 'panoptes':

            my_project = panoptes.init_panoptes()
            # get classification & subject data via Panoptes client
            # cls = panoptes.get_classifications(panoptes.my_project)
            subs = panoptes.get_subject_info(my_project)

            subject_set = SubjectSet(labels=self.classes)

            for key, value in subs.items():
                subject = Subject(identifier=key,
                                  label=value['metadata']['#label'],
                                  meta_data=value['metadata'],
                                  urls=value['url']
                                  )
                subject_set.addSubject(key, subject)

        if mode == 'disk':
            file = self.cfg_path['db'] + 'subject_set.pkl'
            logging.info("Loading %s" % file)
            subject_set = pickle.load(open(file, 'rb'))
            logging.info("Finished Loading %s" % file)

        self.subject_set = subject_set

    def saveSubjectSetOnDisk(self):
        """ Save all Subjects in class specific folders """

        # to retry saving in case of connection errors while fetching urls
        counter = 0
        success = False
        n_trials = 1
        while ((not success) & (counter < n_trials)):
            try:
                self.subject_set.saveOnDisk(set_name='all',
                                            cfg=self.cfg, cfg_path=self.cfg_path)
                success = True
            except Exception as e:
                # log exception
                logging.exception("saveSubjectSetOnDisk failed")
                counter += 1
                print("Failed to Save Subjects on Disk")
                print("Starting attempt %s / %s" % (counter, n_trials))
        if not success:
            IOError("Could not save subject set on disk")


