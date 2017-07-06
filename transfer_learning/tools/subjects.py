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

    def printLabelDistribution(self):
        """ print distribution of labels to stdout """
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

        print("Label Distribution")
        logging.info("Label distribution")
        for r in res_sort:
            print("%s: %s" % (r[0], r[1]))
            logging.info("%s: %s" % (r[0], r[1]))

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
        print("------------------------------------------")
        print("Saving %s data ...." % set_name)
        logging.info("Saving %s data ...." % set_name)
        time_s = time.time()
        urls, labels, ids, fnames = self.getAllURLsLabelsIDsFnames()
        res = data_loader.storeOnDisk(urls=urls,
                                      labels=labels,
                                      fnames=fnames,
                                      path=cfg_path['images'] + set_name,
                                      target_size=cfg['image_size_save'][0:2],
                                      chunk_size=100,
                                      zooniverse_imgproc=False)

        print("Finished saving on disk after %s minutes" %
              ((time.time() - time_s) // 60))
        print("------------------------------------------")
        logging.info("Finished saving on disk after %s minutes" %
                     ((time.time() - time_s) // 60))

        # remove unsuccessful savings
        failures = res['failures']

        # remove whole subject
        for key in failures.keys():
            sub_id = key.split('_')[0]
            self.removeSubject(sub_id)
            # log warning
            logging.warn("Removing filed subject id: %s \n" % sub_id)

        logging.warn("Removing %s of %s subjects" % (len(failures.keys()),
                                                     len(self.getAllIDs()) +
                                                     len(failures.keys())))

        # update path information for all imges in in Subject set
        path = res['path']

        for sub_id in self.getAllIDs():
            sub = self.getSubject(sub_id)
            imgs = sub.getImages()
            label = sub.getLabel()
            for img in imgs.values():
                try:
                    img.setPath(path + label + "/")
                except:
                    print("lalala")

    def save(self, path):
        """ save subject set to disk as json file """
        # get all ids and prepare dictionary
        ids = self.getAllIDs()
        res = dict()

        # loop through subjects and create dict for each subject
        for i in ids:
            s = self.getSubject(i)
            sub_d = {'label': s.getLabel(),
                     'urls': s.getURLs(),
                     'fnames': s.getFileNames(),
                     'file_paths': s.getFilePaths(),
                     'img_ids': list(s.getImages().keys()),
                     'meta_data': s.getMetaData()}
            res[i] = sub_d

        # write to json file
        with open(path, 'w') as fp:
            json.dump(res, fp)

        print("SubjectSet saved to %s" % path)
        logging.info("SubjectSet saved to %s" % path)

    def load(self, path):
        """ re-create subject set from csv / save operation """
        # open file
        file = open(path, 'r')
        res = json.load(file)

        for k, v in res.items():
            # create subject
            s = Subject(identifier=k, label=v['label'],
                        meta_data=v['meta_data'], urls=v['urls'])
            imgs = s.getImages()

            for img, p in zip(list(imgs.keys()), v['file_paths']):
                imgs[img].setPath(p)

            self.addSubject(k, s)

        print("SubjectSet %s Loaded" % path)
        logging.info("SubjectSet %s Loaded" % path)


class Subject(object):
    """ Subject definition """
    def __init__(self, identifier, label, meta_data=None, urls=None):
        self.identifier = str(identifier)
        self.label = label
        self.meta_data = meta_data
        self.images = dict()

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
                self.images[idd] = img

    def getLabel(self):
        return self.label

    def getURLs(self):
        return self.urls

    def getImage(self, fname):
        return self.iamges[fname]

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

    def setLabel(self, label):
        """ Set label """
        self.label = label

    def checkImageFiles(self):
        """ Check if all images are stored on disk """
        for img in self.getImages().values():
            if not img.checkFileExistence():
                return False
        return True

    def getMetaData(self):
        return self.meta_data


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

        if path is None:
            return False
        else:
            return os.path.isfile(path)
