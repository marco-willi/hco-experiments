from tools.subjects import SubjectSet
from sklearn.model_selection import train_test_split
import os
import shutil
from tools.project import Project
from tools.model import Model
from config.config import cfg_path, cfg_model, config
from learning.helpers import create_class_mappings
from config.config import logging


class Experiment(object):
    """ Defines an experiment which is defined by its project, class mapping
        and a model"""

    def __init__(self, name, project, class_list=None, class_mapper=None,
                 train_size=0.9):
        # experiment name
        self.name = name
        # a project object
        self.project = project
        # proportion of training size
        self.train_size = train_size
        # a list of classes to use (better use class mapping)
        self.class_list = class_list
        # a class mapping (preferred option)
        self.class_mapper = class_mapper
        self.classes = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.model = None

        # check input
        if (self.class_list is not None) & (self.class_mapper is not None):
            raise Exception("Either class_mapper or class_list\
                            have to be None")

        # add classes
        if class_list is not None:
            self.classes = class_list
        elif self.class_mapper is not None:
            classes_all = set()
            for value in self.class_mapper.values():
                classes_all.add(value)
            self.classes = list(classes_all)
        self.classes.sort()

    def _classMapper(self, ids, labels):
        """ Map Classes """

        # prepare result lists
        labels_final = list()
        ids_final = list()

        # loop over all labels and ids
        for label, i in zip(labels, ids):
            if self.class_list is not None:
                if label in self.class_list:
                    ids_final.append(i)
                    labels_final.append(label)
            elif self.class_mapper is not None:
                if label in self.class_mapper:
                    new_label = self.class_mapper[label]
                    labels_final.append(new_label)
                    ids_final.append(i)
        return ids_final, labels_final

    def _preparePaths(self, tag, clear_old_files):
        """ prepare paths to save training data """

        # create directories
        root_path = cfg_path['images'] + tag

        # delete old files
        if clear_old_files:
            if os.path.exists(root_path):
                shutil.rmtree(root_path)

        # create root directory
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        # create class directories
        for cl in self.classes:
            if not os.path.exists(root_path + "/" + cl):
                os.mkdir(root_path + "/" + cl)

        return root_path


    def createExpDataSet(self, link_only=True, new_split=True,
                         clear_old_files=True):
        """ Create Test / Train / Validation Data set, if link_only is True
            only symbolik links are created, no actual data is copied """

        # create splits
        if new_split:
            self.createTrainTestSplit()

        for tag, sub_set in zip(['train', 'test', 'val'],
                                [self.train_set, self.test_set, self.val_set]):

            # prepare subject set
            logging.info("Preparing Paths for train/test/val")
            root_path = self._preparePaths(tag, clear_old_files)


            if link_only:
                logging.info("Creating link only files")
                subject_ids = sub_set.getAllIDs()
                for s_i in subject_ids:
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs.values():
                        img.createSymLink(dest_path=root_path + "/" +
                                          label + "/")
            else:
                subject_ids = sub_set.getAllIDs()
                logging.info("Creating hard copy files")
                for s_i in subject_ids:
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs.values():
                        img.copyTo(dest_path=root_path + "/" +
                                   label + "/")

    def createTrainTestSplit(self):
        """ create Test / Train / Validation splits """

        # get random seed
        rand = self.project.cfg['random_seed']

        # get all subject ids and their labels
        ids, labels = self.project.subject_set.getAllIDsLabels()

        # map labels & keep only relevant ids
        ids, labels = self._classMapper(ids, labels)

        # create id to label mapper
        class_mapper_id = dict()
        for i, l in zip(ids, labels):
            class_mapper_id[i] = l

        # training and test split
        id_train, id_test = train_test_split(list(ids),
                                             train_size=self.train_size,
                                             stratify=labels,
                                             random_state=int(rand))

        # validation split
        id_test, id_val = train_test_split(id_test,
                                           train_size=0.5,
                                           random_state=int(rand))

        # generate new subject sets
        train_set = SubjectSet(labels=self.classes)
        test_set = SubjectSet(labels=self.classes)
        val_set = SubjectSet(labels=self.classes)

        set_ids = [id_train, id_test, id_val]
        sets = [train_set, test_set, val_set]
        for si, s in zip(set_ids, sets):
            for i in si:
                sub = self.project.subject_set.getSubject(i)
                # change label
                new_label = class_mapper_id[i]
                sub.overwriteLabel(new_label)
                s.addSubject(i, sub)

        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

    def addModel(self, model):
        """ add a model to the experiment """
        self.model = model

    def train(self):
        """ train model """
        self.model.train()


if __name__ == '__main__':

#    [4715]
#subject_mode: panoptes
#image_storage: disk
##model_file: cat_vs_dog_url
#model_file: cat_vs_dog_disk
#identifier: cat_vs_dog
#classes: cat,dog
#random_seed: 3345
#image_size_save: 200,200,3
#image_size_model: 150,150,3
#num_epochs: 5
#num_classes: 2
#batch_size: 128
#data_augmentation: False
#batch_size_big: 500

    project_id = config['projects']['panoptes_id']
    classes = cfg_model['classes']

    project = Project(name=cfg_model['identifier'],
                      panoptes_id=int(project_id),
                      classes=classes,
                      cfg_path=cfg_path,
                      config=config)

    project.createSubjectSet(mode="panoptes")
    project.saveSubjectSetOnDisk()

    exp = Experiment(name="mnist", project=project,
                     class_list=classes,
                     train_size=0.9)


#    exp = Experiment(name="mnist", project=project,
#                     #class_list=classes,
#                     class_mapper = {'0': 'roundish',
#                                     '1': 'straight',
#                                     '2': 'mixed',
#                                     '3': 'roundish',
#                                     '4': 'straight',
#                                     '5': 'mixed',
#                                     '6': 'roundish',
#                                     '7': 'straight',
#                                     '8': 'roundish',
#                                     '9': 'roundish'},
#                     train_size=0.9)

#    exp = Experiment(name="mnist", project=project,
#                 #class_list=classes,
#                 class_mapper = {'0': 'roundish',
#                                 '1': 'straight'},
#                 train_size=0.9)

    exp.createExpDataSet(link_only=False)

    class_mapper = create_class_mappings(cfg_model['class_mapping'])

    # create model object
    model = Model(train_set=exp.train_set,
                  test_set=exp.test_set,
                  val_set=exp.val_set,
                  mod_file=cfg_model['model_file'],
                  pre_processing=cfg_model['pre_processing'],
                  config=config,
                  cfg_path=cfg_path,
                  callbacks=cfg_model['callbacks'],
                  optimizer=cfg_model['optimizer'],
                  num_classes=len(class_mapper.keys()))

    exp.addModel(model)

    exp.train()





#
#    project = Project(name="cats_vs_dogs", panoptes_id=4715,
#                      classes=['cat','dog'],
#                      cfg_path=cfg_path,
#                      config=config)
#
#    project.createSubjectSet(mode="panoptes")
#    project.saveSubjectSetOnDisk()
#
#
#    exp = Experiment(name="cats_vs_dogs_V1", project=project,
#                     classes="all")
