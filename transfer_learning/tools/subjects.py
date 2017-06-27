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


class SubjectSet(object):
    """ Defines a set of subjects """
    def __init__(self, labels):
        self.labels = labels
        self.subjects = dict()
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        self.labels_num = self.le.transform(self.labels)

    def addSubject(self, subject_id, subject):
        self.subjects[str(subject_id)] = subject

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
            ids.append(i)
            labels.append(self.subjects[i].getLabel())

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
            for url in sub.getURLs():
                labels.append(sub.getLabel())
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
            for url in sub.getURLs():
                labels.append(sub.getLabel())
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
            for url, fname in zip(sub.getURLs(), sub.getFileNames()):
                labels.append(sub.getLabel())
                urls.append(url)
                ids.append(i)
                fnames.append(fname)
        return urls, labels, ids, fnames

    def saveOnDisk(self, set_name, cfg, cfg_path):
        """ save all subjects to disk """

        # invoke bulk reading
        data_loader = ImageUrlLoader()

        # save to disk
        print("------------------------------------------")
        print("Saving %s data ...." % set_name)
        time_s = time.time()
        urls, labels, ids, fnames = self.getAllURLsLabelsIDsFnames()
        res = data_loader.storeOnDisk(urls=urls,
                                       labels=labels,
                                       fnames=fnames,
                                       path=cfg_path['images'] + set_name,
                                       target_size=cfg['image_size_save'][0:2],
                                       chunk_size=100)

        print("Finished saving on disk after %s minutes" %
              ((time.time() - time_s) // 60))
        print("------------------------------------------")

        # remove unsuccessful savings
        failures = res['failures']

        # remove whole subject
        for key in failures.keys():
            sub_id = key.split('_')[0]
            self.removeSubject(sub_id)

        # update path information for all imges in in Subject set
        path = res['path']

        for sub_id in set(ids).difference(failures.keys()):
            sub = self.getSubject(sub_id)
            imgs = sub.getImages()
            label = sub.getLabel()
            for img in imgs:
                img.setPath(path + label + "/")


class Subject(object):
    """ Subject definition """
    def __init__(self, identifier, label, meta_data=None, urls=None):
        self.identifier = str(identifier)
        self.label = label
        self.meta_data = meta_data
        self.images = list()

        # handle urls
        if isinstance(urls, list):
            self.urls = urls
        else:
            self.urls = [urls]
        # sort urls in reverse order
        self.urls = [x[::-1] for x in sorted([x[::-1] for x in self.urls])]

        # create image objects
        if urls is not None:
            for i in range(0, len(self.urls)):
                # idd = np.array_str(self.identifier) + '_' + str(i)
                idd = self.identifier + '_' + str(i)
                img = Image(identifier=idd,
                            url=self.urls[i])
                self.images.append(img)

    def getLabel(self):
        return self.label

    def getURLs(self):
        return self.urls

    def _setURLs(self, urls):
        self.urls = [x[::-1] for x in sorted([x[::-1] for x in urls])]

    def setMetaData(self, meta_data):
        self.meta_data = meta_data

    def getFileNames(self):
        fnames = list()
        for img in self.images:
            fnames.append(img.getFilename())
        return fnames

    def getImages(self):
        return self.images

    def overwriteLabel(self, label):
        self.label = label


class Image(object):
    """ Defines a single image, which is part of a subject """
    def __init__(self, identifier, url=None, path=None):
        self.url = url
        self.path = path
        self.identifier = identifier
        self.filename = identifier + ".jpeg"

    def setPath(self, path):
        self.path = path

    def getPath(self):
        return self.path

    def setURL(self, url):
        self.url = url

    def createSymLink(self, dest_path):
        os.symlink(self.path + self.filename, dest_path + self.filename)

    def getFilename(self):
        return self.filename

    def copyTo(self, dest_path):
        shutil.copyfile(src=self.path + self.filename,
                        dst=dest_path + self.filename)



