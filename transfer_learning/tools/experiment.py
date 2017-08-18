from tools.subjects import SubjectSet
from sklearn.model_selection import train_test_split
import os
import shutil
from tools.project import Project
from tools.model import Model
from tools.helpers import createSplitIDs
from config.config import cfg_path, cfg_model, config
from learning.model_components import create_class_mappings
from config.config import logging
import random


class Experiment(object):
    """ Defines an experiment which is defined by its project, class mapping
        and a model"""

    def __init__(self, name, project, class_list=None, class_mapper=None,
                 train_size=0.9, test_size=None, equal_class_sizes=False,
                 random_state=123,
                 max_labels_per_subject=None):
        # experiment name
        self.name = name
        # a project object
        self.project = project
        # proportion of training size
        self.train_size = train_size
        self.test_size = test_size
        # a list of classes to use (better use class mapping)
        self.class_list = class_list
        # a class mapping (preferred option)
        self.class_mapper = class_mapper
        # random state if required
        self.random_state = random_state
        # class sizes
        self.equal_class_sizes = equal_class_sizes
        # max allowed labels per subject
        self.max_labels_per_subject = max_labels_per_subject
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
        logging.debug("ClassMapper contains %s ids and %s labels" %
                      (len(ids_final), len(labels_final)))

        return ids_final, labels_final

    def _preparePaths(self, tag, clear_old_files):
        """ prepare paths to save training data """

        # create directories
        root_path = cfg_path['images'] + tag

        # delete old files
        if clear_old_files:
            if os.path.exists(root_path):
                logging.debug("Deleting %s" % root_path)
                shutil.rmtree(root_path)

        # create root directory
        if not os.path.exists(root_path):
            logging.debug("Creating %s" % root_path)
            os.mkdir(root_path)

        # delete all non-relevant class directories
        if not clear_old_files:
            all_class_dirs = os.listdir(root_path)
            delete_dirs = set(all_class_dirs) - set(self.classes)
            if len(delete_dirs) > 0:
                for d in delete_dirs:
                    logging.debug("Removing directory %s" %
                                  (root_path + os.path.sep + d))
                    shutil.rmtree(root_path + os.path.sep + d)

        # create class directories
        for cl in self.classes:
            if not os.path.exists(root_path + os.path.sep + cl):
                logging.debug("Creating %s" % (root_path + os.path.sep + cl))
                os.mkdir(root_path + os.path.sep + cl)

        return root_path

    def save(self):
        """ Save test/train/val datasets to disk """
        # check whether they are defined or not
        if any(x is None for x in [self.test_set, self.train_set,
                                   self.val_set]):
            raise IOError("test, train, val sets are None")

        # save subject sets on disk
        for s, n in zip([self.test_set, self.train_set, self.val_set],
                        ['test', 'train', 'val']):
            path = self.project.cfg_path['db'] + n + '_subject_set_' +\
                   self.name + '.json'

            s.save(path)

            # save aggregate statistics
            path_csv = self.project.cfg_path['db'] + n + '_subject_set_' +\
                self.name + '_label_dist_subjects.csv'
            s.to_csv(path_csv, mode="label_dist_subjects")

            path_csv = self.project.cfg_path['db'] + n + '_subject_set_' +\
                self.name + '_label_dist_images.csv'
            s.to_csv(path_csv, mode="label_dist_images")

    def load(self):
        """ Load test/tain/val datasets from disk """

        # create subject sets
        self.train_set = SubjectSet(labels=self.classes)
        self.test_set = SubjectSet(labels=self.classes)
        self.val_set = SubjectSet(labels=self.classes)

        # load subject sets from disk
        for s, n in zip([self.test_set, self.train_set, self.val_set],
                        ['test', 'train', 'val']):
            path = self.project.cfg_path['db'] + n + '_subject_set_' +\
                   self.name + '.json'
            # abort if one file is missing and raise exception
            if not os.path.isfile(path):
                raise FileExistsError("File %s does not exist" % path)
            s.load(path)

    def createExpDataSet(self, link_only=True,
                         splits="new",
                         clear_old_files=True,
                         split_mode="none"):
        """ Create Test / Train / Validation Data set, if link_only is True
            only symbolik links are created, no actual data is copied """

        # create new splits
        if splits == 'new':
            self.createTrainTestSplit(save_to_disk=True, split_mode=split_mode)
        elif splits == 'disk':
            # if files are there load them from disk, else create new
            try:
                self.load()
            except:
                self.createTrainTestSplit(save_to_disk=True,
                                          split_mode=split_mode)
        else:
            raise NotImplementedError

        logging.debug("Starting to prepare experiment datasets")
        for tag, sub_set in zip(['train', 'test', 'val'],
                                [self.train_set, self.test_set, self.val_set]):

            # prepare subject set
            logging.debug("Preparing Paths for %s" % tag)
            root_path = self._preparePaths(tag, clear_old_files)

            # get all relevant subject ids
            subject_ids = sub_set.getAllIDs()

            # if max labels per subject is restricted remove all
            # that exceed that threshold
            if self.max_labels_per_subject is not None:
                subject_ids_allowed = list()
                for ii in subject_ids:
                    n_labels = sub_set.getSubject(ii).getNumLabels()
                    if n_labels <= self.max_labels_per_subject:
                        subject_ids_allowed.append(ii)
                subject_ids = subject_ids_allowed

            # check if some already exist and keep them
            if not clear_old_files:
                # get all files already on disk
                all_classes = os.listdir(root_path)

                # store information of existing files in dictionary
                existing_dict = dict()
                for cl in all_classes:
                    existing_files = os.listdir(root_path + os.path.sep + cl +
                                                os.path.sep)
                    for ex in existing_files:
                        existing_id = ex.split('_')[0]
                        if existing_id not in existing_dict:
                            existing_dict[existing_id] = {'cl': cl,
                                                          'files': list()}
                        existing_dict[existing_id]['files'].append(ex)

                if len(existing_dict.keys()) == 0:
                    logging.debug("No files exist in %s directory" % tag)
                else:
                    logging.debug("%s files already exist in %s directory" %
                                  (len(existing_dict.keys()), tag))
                    # relevant subject ids that are not already on disk
                    subject_ids_relev = set(subject_ids) - existing_dict.keys()

                    # existing files that have to be removed
                    to_be_removed = existing_dict.keys() - set(subject_ids)

                    # remove files
                    for r in to_be_removed:
                        files_to_remove = existing_dict[r]['files']
                        class_to_be_removed = existing_dict[r]['cl']
                        for fr in files_to_remove:
                            os.remove(root_path + os.path.sep +
                                      class_to_be_removed +
                                      os.path.sep + fr)

                    # only keep subject ids that are not already on disk
                    subject_ids = list(subject_ids_relev)

            if link_only:
                logging.info("Creating link only files")

                for s_i, c in zip(subject_ids, range(0, len(subject_ids))):
                    if (c % 10000) == 0:
                        logging.debug("Link %s / %s created" %
                                      (c, len(subject_ids)))
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs.values():
                        img.createSymLink(dest_path=root_path + os.path.sep +
                                          label + os.path.sep)
            else:
                logging.info("Creating hard copy files")
                for s_i in subject_ids:
                    sub = sub_set.getSubject(s_i)
                    imgs = sub.getImages()
                    label = sub.getLabel()

                    for img in imgs.values():
                        img.copyTo(dest_path=root_path + os.path.sep +
                                   label + os.path.sep)

    def _balancedSampling(self, ids, labels):
        """ downsample larger classes to match size of smallest class """

        logging.info("Balanced class sampling ...")

        # count labels
        label_count = dict()
        for l in labels:
            if l not in label_count:
                label_count[l] = 1
            else:
                label_count[l] += 1

        # build class mapper
        class_map = dict()
        for i, l in zip(ids, labels):
            if l not in class_map:
                class_map[l] = list()
            class_map[l].append(i)

        # smallest class size
        smallest = min(list(label_count.values()))

        # randomly select ids of larger classes
        ids_final = list()
        labels_final = list()

        for l, i in class_map.items():
            # sample classes
            if len(i) > smallest:
                random.seed(self.random_state)
                i_sampled = random.sample(population=i, k=smallest)
            else:
                i_sampled = i
            # add to output
            ids_final.extend(i_sampled)
            labels_final.extend([l for x in range(0, len(i_sampled))])

        return ids_final, labels_final

    def createTrainTestSplit(self, save_to_disk=False, split_mode="none"):
        """ create Test / Train / Validation splits """

        # get random seed
        rand = self.random_state

        # get all subject ids and their labels
        ids, labels = self.project.subject_set.getAllIDsLabels()

        # prepare meta data dictionary for all subjects
        meta_data = dict()
        for i in ids:
            meta_data[i] = self.project.subjec_set.getSubject(i).getMetaData()

        # create splitting id to split subjects on, using original, unmapped
        # labels
        ids_orig, split_ids, split_labels =\
            createSplitIDs(ids, labels, meta_data=meta_data,
                           split_mode=split_mode)

        # map labels to classes & keep only relevant ids
        ids, labels = self._classMapper(ids, labels)

        # if equal class sizes, cut larger classes to size of smallest
        if self.equal_class_sizes:
            ids, labels = self._balancedSampling(ids, labels)

        # create id to label mapper
        class_mapper_id = dict()
        for i, l in zip(ids, labels):
            class_mapper_id[i] = l

        # create split id to label mapper
        class_mapper_split_id = dict()
        for i, l in zip(split_ids, split_labels):
            class_mapper_split_id[i] = l

        # mapper split ids to orig ids
        split_id_mapper = dict()
        for jj in range(0, len(split_ids)):
            if split_ids[jj] not in split_id_mapper:
                split_id_mapper[split_ids[jj]] = [ids_orig[jj]]
            else:
                split_id_mapper[split_ids[jj]].append(ids_orig[jj])

        # mapper orig id to split id
        id_to_split_id_mapper = dict()
        for k, v in split_id_mapper.items():
            for i in v:
                id_to_split_id_mapper[i] = k

        # get rid of all split ids of ids which have been removed by
        # class mapper and balanced sampling
        split_ids = [id_to_split_id_mapper[i] for i in ids]
        split_labels = [class_mapper_split_id[i] for i in split_ids]

        # deduplicate splitting ids to be used in creating test / train splits
        split_ids_dedup, split_labels_dedup = list(), list()
        for k, v in class_mapper_split_id.items():
            split_ids_dedup.append(k)
            split_labels_dedup.append(v)

        # training and test split
        id_train_s, id_test_s = train_test_split(split_ids_dedup,
                                                 train_size=self.train_size,
                                                 test_size=self.test_size,
                                                 stratify=split_labels_dedup,
                                                 random_state=int(rand))

        # validation split
        labels_s_val = [class_mapper_split_id[x] for x in id_test_s]
        id_test_s, id_val_s = train_test_split(id_test_s,
                                               train_size=0.5,
                                               stratify=labels_s_val,
                                               random_state=int(rand))

        # map split ids to original ids
        id_train = [[x for x in split_id_mapper[i]] for i in id_train_s]
        id_test = [[x for x in split_id_mapper[i]] for i in id_test_s]
        id_val = [[x for x in split_id_mapper[i]] for i in id_val_s]

        # get rid of sublists
        id_train = [item for sublist in id_train for item in sublist]
        id_test = [item for sublist in id_test for item in sublist]
        id_val = [item for sublist in id_val for item in sublist]

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
                sub.setLabel(new_label)
                s.addSubject(i, sub)

        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set

        # save subject sets on disk
        if save_to_disk:
            self.save()

        # print label distribution
        for s, l in zip([self.train_set, self.test_set, self.val_set],
                        ['train', 'test', 'val']):
            logging.info("Label Distribution %s" % l)
            s.printLabelDistribution()

    def addModel(self, model):
        """ add a model to the experiment """
        self.model = model

    def prep_model(self):
        """ prepare model """
        try:
            self.model.prep_model()
        except Exception as e:
            # log exception
            logging.exception("model preparation failed")
            raise Exception

    def train(self):
        """ train model """
        try:
            self.model.train()
        except Exception as e:
            # log exception
            logging.exception("model training failed")
            raise Exception

    def evaluate(self):
        """ evaluate model """
        logging.info("evaluating model")
        self.model.evaluate()


if __name__ == '__main__':

    project_id = config['projects']['panoptes_id']
    classes = cfg_model['classes']

    project = Project(name=str(project_id),
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
