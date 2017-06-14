from tools import panoptes
from tools.subjects import SubjectSet, Subject
from tools.save_images_on_disk import save_on_disk
import pickle
from config.config import config, cfg_path
from sklearn.model_selection import train_test_split
import os
from tools.model_helpers import model_param_loader
import shutil
import importlib


class Project(object):
    """ class to implement a Zooniverse project """
    def __init__(self, name, panoptes_id,
                 classes, cfg_path, config):
        self.name = name
        self.panoptes_id = panoptes_id
        self.subject_set = None
        self.classes = classes
        self.cfg_path = cfg_path
        self.cfg = model_param_loader(config)

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
                                  urls=value['url'],
                                  label_num=subject_set.getLabelEncoder().
                                  transform([value['metadata']['#label']])
                                  )
                subject_set.addSubject(key, subject)

        if mode == 'disk':
            file = self.cfg_path['db'] + 'subject_set.pkl'
            subject_set = pickle.load(open(file, 'rb'))

        self.subject_set = subject_set

    def saveSubjectSetOnDisk(self):
        """ Save all Subjects in class specific folders """

        self.subject_set.saveOnDisk(set_name='all',
                                    cfg=self.cfg, cfg_path=self.cfg_path)


class Experiment(object):
    """ Defines an experiment """
    def __init__(self, name, project, class_list=None, class_mapper=None,
                 train_size=0.9):
        self.name = name
        self.project = project
        self.train_size = train_size
        self.class_list = class_list
        self.class_mapper = class_mapper
        self.classes = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.model_file = None

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
        """ Create Test / Train / Validation Data set """

        # create splits
        if new_split:
            self.createTrainTestSplit()

        for tag, sub_set in zip(['train', 'test', 'val'],
                                [self.train_set, self.test_set, self.val_set]):

            # prepare subject set
            root_path = self._preparePaths(tag, clear_old_files)


            if link_only:
                subject_ids = sub_set.getAllIDs()
                for s_i in subject_ids:
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs:
                        img.createSymLink(dest_path = root_path + "/" +
                                          label + "/")
            else:
                subject_ids = sub_set.getAllIDs()
                for s_i in subject_ids:
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs:
                        img.copyTo(dest_path = root_path + "/" +
                                   label + "/")

    def createTrainTestSplit(self):
        """ create Test / Train / Validation splits """

        # get random seed
        rand = self.project.cfg['random_seed']

        # get all subject ids and their labels
        ids, labels = project.subject_set.getAllIDsLabels()

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
                sub = project.subject_set.getSubject(i)
                # change label
                new_label = class_mapper_id[i]
                sub.overwriteLabel(new_label)
                s.addSubject(i, sub)

        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

    def addModelFile(self, file):
        """ define a model file in models submodule """
        self.model_file = file

    def train(self):
        """ train model """
        # import model file
        model = importlib.import_module('models.' + self.model_file)

        # train model
        model.train(self.train_set, self.test_set, self.val_set)



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
    classes = config[project_id]['classes'].replace("\n","").split(",")

    project = Project(name=config[project_id]['identifier'],
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

    exp.addModelFile(config[project_id]['model_file'])

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

