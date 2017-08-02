"""
Class to implement a Project
- defines a set of images and their classes to train a model from
- typically this would be a single Zooniverse project but can also be
  a collection of several projects
"""
from tools import panoptes
from tools.subjects import SubjectSet, Subject
from config.config import cfg_model as cfg
from config.config import logging
import os


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

    def loadSubjectSet(self, file):
        """ Load Subject Set from Disk """
        # create empty subject set
        self.subject_set = SubjectSet(labels=self.classes)

        # load subject data from json file
        self.subject_set.load(file)

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

            # save subject set on disk
            subject_set.save(self.cfg_path['db'] + 'subject_set.json')
            self.subject_set = subject_set

        # load from disk
        if mode == 'disk':
            file = self.cfg_path['db'] + 'subject_set.json'
            logging.info("Loading %s" % file)
            self.loadSubjectSet(file)
            logging.info("Finished Loading %s" % file)

        if mode == 'disk_used':
            file = self.cfg_path['db'] + 'subject_set_used.json'
            if os.path.isfile(file):
                logging.info("Loading %s" % file)
                self.loadSubjectSet(file)
                logging.info("Finished Loading %s" % file)
            else:
                file = self.cfg_path['db'] + 'subject_set.json'
                logging.info("Loading %s" % file)
                self.loadSubjectSet(file)
                logging.info("Finished Loading %s" % file)

    def saveSubjectSetOnDisk(self, overwrite=True):
        """ Save all Subjects / images in class specific folders """

        # check if subject set is already on disk
        if not overwrite:
            file_path = self.cfg_path['db'] + 'subject_set_used.json'
            if os.path.isfile(path=file_path):
                logging.debug("Subjec %s found, not gonna overwrite" %
                              file_path)
                return None
        # to retry saving in case of connection errors while fetching urls
        counter = 0
        success = False
        n_trials = 99
        while ((not success) & (counter < n_trials)):
            try:
                self.subject_set.saveImagesOnDisk(set_name='all',
                                                  cfg=self.cfg,
                                                  cfg_path=self.cfg_path)
                success = True
            except Exception as e:
                # log exception
                logging.exception("saveSubjectSetOnDisk failed")
                counter += 1
                logging.info("Starting attempt %s / %s" % (counter, n_trials))
        if not success:
            IOError("Could not save subject set on disk")
        else:
            # remove unsuccessfully processed subjects
            self.subject_set.removeSubjectsWithoutAllImages()

            # save subject set containing only successfully processed
            # subjects to disk
            self.subject_set.save(path=self.cfg_path['db'] +
                                  'subject_set_used.json')
            logging.info("Saved %s to disk" % (self.cfg_path['db'] +
                         'subject_set_used.json'))

            # print label distribution
            self.subject_set.printLabelDistribution()
