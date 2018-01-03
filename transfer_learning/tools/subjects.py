"""
Classes to implement SubjectSets, Subjects and Images
- A SubjectSet is a collection of Subjects which typically are either
  the full subject set of a project, a training-, test- or validation set.
- A Subject is defined by a unique identifier, by a label and by a set
  of images.
- An Image belongs to a Subject, and has an URL and possibly a location on disk
"""
from sklearn.preprocessing import LabelEncoder
import os
from tools.image_url_loader import ImageUrlLoader
import time
import shutil
from config.config import logging
import json
from datetime import datetime


class SubjectSet(object):
    """ Defines a set of subjects """
    def __init__(self, labels):
        self.labels = labels
        self.subjects = dict()
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        self.labels_num = self.le.transform(self.labels)

    def addSubject(self, subject):
        self.subjects[subject.getID()] = subject

    def getSubject(self, subject_id):
        return self.subjects[str(subject_id)]

    def removeSubject(self, subject_id):
        self.subjects.pop(str(subject_id), None)

    def getLabelEncoder(self):
        return self.le

    def getAllIDs(self):
        # return sorted ids
        ids = sorted(self.subjects.keys())
        return ids

    def getAllIDsLabels(self):

        # prepare results
        ids = list()
        labels = list()

        # get all ids
        all_ids = self.getAllIDs()

        # iterate over all ids and get label
        for i in all_ids:
            # get all labels
            labs_sub = self.subjects[i].getLabels(mode="concat")
            if type(labs_sub) is str:
                labs_sub = [labs_sub]
            for label in labs_sub:
                ids.append(i)
                labels.append(label)

        return ids, labels

    def getAllURLsLabels(self):
        """ return all URLS and corresponding labels """
        # prepare results
        urls = list()
        labels = list()

        # get all ids
        all_ids = self.getAllIDs()

        for i in all_ids:
            sub = self.subjects[i]
            # get all labels
            labs_sub = sub.getLabels(mode="concat")
            if type(labs_sub) == str:
                labs_sub = [labs_sub]
            for label in labs_sub:
                # get all urls
                for url in sub.getURLs():
                    labels.append(label)
                    urls.append(url)
        return urls, labels

    def getAllURLsLabelsIDs(self):
        """ return all URLS and corresponding labels """
        # prepare results
        urls = list()
        labels = list()
        ids = list()

        # get all ids
        all_ids = self.getAllIDs()

        for i in all_ids:
            sub = self.subjects[i]
            # get all labels
            labs_sub = sub.getLabels(mode="concat")
            if type(labs_sub) is str:
                labs_sub = [labs_sub]
            for label in labs_sub:
                # get all urls
                for url in sub.getURLs():
                    labels.append(label)
                    urls.append(url)
                    ids.append(i)
        return urls, labels, ids

    def getAllURLsLabelsIDsFnames(self):
        """ return all URLS and corresponding labels """
        # prepare results
        urls = list()
        labels = list()
        ids = list()
        fnames = list()

        # get all ids
        all_ids = self.getAllIDs()

        for i in all_ids:
            sub = self.subjects[i]
            # get all labels
            labs_sub = sub.getLabels(mode="concat")
            if type(labs_sub) is str:
                labs_sub = [labs_sub]
            for label in labs_sub:
                # get all urls
                for url, fname in zip(sub.getURLs(), sub.getFileNames()):
                    labels.append(label)
                    urls.append(url)
                    ids.append(i)
                    fnames.append(fname)
        return urls, labels, ids, fnames

    def printLabelDistribution(self):
        """ print distribution of labels to stdout / logging module """
        # get all labels
        ids, labels = self.getAllIDsLabels()

        # count labels
        res = dict()
        for lab in labels:
            if lab not in res.keys():
                res[lab] = 1
            else:
                res[lab] += 1

        # sort labels
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)

        logging.info("Label distribution")
        for r in res_sort:
            logging.info("{:15s} {:7d} / {:7d} {:.2%}".format(r[0],
                                                              r[1], len(ids),
                                                              int(r[1]) /
                                                              len(ids)))

    def getSubjectsWithoutAllImages(self):
        """ return subject ids without all images on disk """
        res = list()
        for s in self.getAllIDs():
            sub = self.getSubject(s)
            if not sub.checkImageFiles():
                res.append(s)
        return res

    def removeSubjectsWithoutAllImages(self):
        """ Remove all subjects that don't have all images on disk """
        failed = self.getSubjectsWithoutAllImages()

        for sub in failed:
            self.removeSubject(sub)

    def saveImagesOnDisk(self, set_name, cfg, cfg_path):
        """ save all subjects to disk """

        # invoke bulk reading
        data_loader = ImageUrlLoader()

        # save to disk
        logging.info("Saving %s data ...." % set_name)
        time_s = time.time()
        urls, labels, ids, fnames = self.getAllURLsLabelsIDsFnames()
        if cfg['image_size_save'] is not None:
            save_size = cfg['image_size_save'][0:2]
        else:
            save_size = None
        res = data_loader.storeOnDisk(urls=urls,
                                      labels=labels,
                                      fnames=fnames,
                                      path=cfg_path['images'] + set_name,
                                      target_size=save_size,
                                      chunk_size=100,
                                      zooniverse_imgproc=False)
        logging.info("Finished saving on disk after %s minutes" %
                     ((time.time() - time_s) // 60))

        # remove unsuccessful savings
        failures = res['failures']

        # remove whole subject
        for key in failures.keys():
            sub_id = key.split('_')[0]
            self.removeSubject(sub_id)
            # log warning
            logging.warn("Removing failed subject id: %s \n" % sub_id)

        logging.warn("Removing %s of %s subjects" % (len(failures.keys()),
                                                     len(self.getAllIDs()) +
                                                     len(failures.keys())))

        # update path information for all imges in in Subject set
        path = res['path']

        for sub_id in self.getAllIDs():
            sub = self.getSubject(sub_id)
            imgs = sub.getImages()
            label = sub.getLabels(mode="concat")
            for img in imgs.values():
                try:
                    img.setPath(path + label + "/")
                except:
                    logging.warning("Failed to set path")

    def to_csv(self, out_file, mode="label_dist_subjects"):
        """ save subject set data to disk as csv """

        # check input
        if out_file is None:
            IOError("out_file must be specified")

        # check mode
        all_modes = ['label_dist_subjects', 'label_dist_images',
                     'all_images', 'all_subjects']
        if mode not in all_modes:
            IOError("mode must be one of: %s" % all_modes)

        # save label distribution of subjects
        if mode in ['label_dist_subjects', 'label_dist_images']:

            # get all labels and subjects / urls
            if mode == 'label_dist_subjects':
                ids, labels = self.getAllIDsLabels()
            elif mode == 'label_dist_images':
                ids, labels = self.getAllURLsLabels()

            # count labels
            res = dict()
            for lab in labels:
                if lab not in res.keys():
                    res[lab] = 1
                else:
                    res[lab] += 1

            # sort labels
            res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)

            # export to csv
            file = open(out_file, "w")
            file.write('class,n\n')
            for r in res_sort:
                # write labels to disk
                file.write(str(r[0]) + ',' + str(r[1]) + "\n")

        if mode in ['all_images']:
            # get all labels, ids, urls and fnames
            urls, labels, ids, fnames = self.getAllURLsLabelsIDsFnames()

            # export to csv
            file = open(out_file, "w")
            file.write('subject_id,label,fname,url\n')

            # loop over all and write to file
            for u, l, i, f in zip(urls, labels, ids, fnames):
                # write labels to disk
                file.write("%s,%s,%s,%s\n" % (i, l, f, u))

    def save(self, path):
        """ save subject set to disk as json file """
        # get all ids and prepare dictionary
        ids = self.getAllIDs()
        res = dict()

        # loop through subjects and create dict for each subject
        for i in ids:
            s = self.getSubject(i)
            sub_d = {'label': s.getLabels(mode="all"),
                     'urls': s.getURLs(),
                     'fnames': s.getFileNames(),
                     'file_paths': s.getFilePaths(),
                     'img_ids': list(s.getImages().keys()),
                     'meta_data': s.getMetaData()}
            res[i] = sub_d

        # write to json file
        with open(path, 'w') as fp:
            json.dump(res, fp)

        logging.info("SubjectSet saved to %s" % path)

    def load(self, path):
        """ re-create subject set from csv / save operation """
        # open file
        file = open(path, 'r')
        res = json.load(file)

        for k, v in res.items():
            # create subject
            s = Subject(identifier=k, labels=v['label'],
                        meta_data=v['meta_data'], urls=v['urls'])
            imgs = s.getImages()

            for img, p in zip(list(imgs.keys()), v['file_paths']):
                imgs[img].setPath(p)

            self.addSubject(s)

        # Log / Print Information
        logging.info("SubjectSet %s Loaded" % path)
        logging.info("Contains %s subjects" % len(self.subjects.keys()))
        self.printLabelDistribution()

    def add_dir_to_set(self, path):
        """ add a directory with images to the subject set """
        # TODO: not yet finished
        # raise NotImplementedError
        # get label of directory
        label = path.split(os.path.sep)[-1]

        # list all files
        files = os.listdir(path)

        # create subjects
        for f in files:
            # identifier
            identifier = f.split('.')[0]
            # create object
            subject = Subject(identifier=identifier, labels=label)
            # add to set
            self.addSubject(subject)


class Subject(object):
    """ Subject definition """
    def __init__(self, identifier, labels, meta_data=None, urls=""):
        self.identifier = str(identifier)
        self.labels = Labels(labels)
        self.n_labels = self.labels.getNumLabels()
        self.meta_data = meta_data
        self.images = dict()
        self.urls = urls

        # handle urls
        if urls is not "":
            if isinstance(urls, list):
                self.urls = urls
            else:
                self.urls = [urls]
            # sort urls in reverse order
            self.urls = [x[::-1] for x in sorted([x[::-1] for x in self.urls])]

            for i in range(0, len(self.urls)):
                # idd = np.array_str(self.identifier) + '_' + str(i)
                idd = self.identifier + '_' + str(i)
                img = Image(identifier=idd,
                            url=self.urls[i])
                self.images[idd] = img

    def getID(self):
        """ get identifier """
        return self.identifier

    def getLabels(self, mode="first"):
        """ Get labels """

        if mode == "first":
            return self.labels.getLabelsFirstOnly()

        elif mode == "all":
            return self.labels.getLabels()

        elif mode == "concat":
            return self.labels.getLabelsOneString()

    def getURLs(self):
        return self.urls

    def getImage(self, fname):
        return self.images[fname]

    def _setURLs(self, urls):
        self.urls = [x[::-1] for x in sorted([x[::-1] for x in urls])]

    def setMetaData(self, meta_data):
        self.meta_data = meta_data

    def getFilePaths(self):
        paths = list()
        for v in self.images.values():
            paths.append(v.getPath())
        return paths

    def getFileNames(self):
        fnames = list()
        for v in self.images.values():
            fnames.append(v.getFilename())
        return fnames

    def getImages(self):
        """ get all images """
        return self.images

    def checkImageFiles(self):
        """ Check if all images are stored on disk """
        for img in self.getImages().values():
            if not img.checkFileExistence():
                return False
        return True

    def getMetaData(self):
        return self.meta_data

    def getNumLabels(self):
        return self.n_labels

    def setLabels(self, labels):
        self.labels = Labels(labels)


class Image(object):
    """ Defines a single image, which is part of a subject """
    def __init__(self, identifier, url=None, path=None):
        self.url = url
        self.path = path
        self.identifier = identifier
        self.filename = identifier + ".jpeg"

    def setPath(self, path):
        """ set path (directory) of image """
        self.path = path

    def getPath(self):
        """ returns path """
        return self.path

    def setURL(self, url):
        """ set URL of image """
        self.url = url

    def createSymLink(self, dest_path):
        """ creates symbolic link on dest_path """
        os.symlink(self.path + self.filename, dest_path + self.filename)

    def getFilename(self):
        """ returns file name including image postfix """
        return self.filename

    def copyTo(self, dest_path):
        """ copy image to """
        shutil.copyfile(src=self.path + self.filename,
                        dst=dest_path + self.filename)

    def checkFileExistence(self):
        """ Check if image file exists """
        path = self.getPath()
        fname = self.getFilename()

        if path is None:
            return False
        else:
            return os.path.isfile(path + fname)


class Labels(object):
    """ Defines the labels of a subject """

    def __init__(self, labels):
        self.labels = None
        self.n_labels = None

        # check input and set labels
        self._setLabels(labels)

    def _setLabels(self, labels):
        """ Set labels and check input """
        # check input
        assert type(labels) in (str, list),\
            "Label is not of type str/list, instead is %s" % type(labels)

        # convert label to list
        if type(labels) is str:
            labels = [labels]

        # set number of labels
        self.n_labels = len(labels)

        self.labels = labels

    def getLabels(self):
        """ Returns list of labels """
        return self.labels

    def getLabelsOneString(self):
        """ Returns one string with concatenated labels """
        return "_".join(self.labels)

    def getLabelsFirstOnly(self):
        """ Returns first label """
        return self.labels[0]

    def getNumLabels(self):
        """ Return number of labels """
        return self.n_labels


if __name__ == "__main__":
    sset = SubjectSet(labels=["monkey", "elephant"])
    s1 = Subject(identifier="1", labels="monkey")
    s2 = Subject(identifier="2", labels="elephant")
    s3 = Subject(identifier="3", labels=["elephant", "monkey"])
    subs = [s1, s2, s3]
    for s in subs:
        sset.addSubject(subject=s)
        print(sset.getSubject(s.getID()).getLabels())
        print(sset.getSubject(s.getID()).getLabels(mode="all"))
        print(sset.getSubject(s.getID()).getLabels(mode="concat"))

    sset.save(path="D:\\Studium_GD\\Zooniverse\Data\\transfer_learning_project\\scratch\\test.json")
    sset2 = SubjectSet(labels=["monkey", "elephant"])
    sset2.load(path="D:\\Studium_GD\\Zooniverse\Data\\transfer_learning_project\\scratch\\test.json")
    sset2.getSubject("3").getLabels()
    sset2.getAllIDsLabels()
