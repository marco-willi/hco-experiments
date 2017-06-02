# import modules
import random
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
import requests
from io import BytesIO
import time
import os
import pickle as pkl
import psutil
from tools.image_url_loader import ImageUrlLoader


# Class to define a Batch of Data
class DataBatch(object):
    """ Class to define a Batch of Data
    - not intended to be directly used by user

    Arguments:
         ids (list): unique ids of all observations
         batch_id (int): unique id of batch
    """
    def __init__(self, ids, batch_id):
        self.ids = ids
        self.y_data = list()
        self.batch_id = batch_id
        self.is_stored = False
        self.batch_subjects = dict()
        self.raw_data = dict()
        self.disk_storage_path = None
        self.storage = None

    def setDiskStoragePath(self, path):
        """ set disk storage path """
        self.disk_storage_path = path

    def getBatchDir(self):
        """ Return directory of all batch subjects """
        return self.batch_subjects

    def getBatchData(self):
        """ Returns numpy arrays of batch data - only if stored"""
        if self.is_stored:
            if self.storage == "memory":
                dat = self.raw_data
                return dat['X_data'], dat['y_data']

            elif self.storage == "disk":
                batch_file = 'batch' + str(self.batch_id) + '.pkl'
                dat = pkl.load(open(self.disk_storage_path + batch_file, "rb"))
                return dat['X_data'], dat['y_data']

            elif self.storage == "numpy":
                batch_file = 'batch' + str(self.batch_id) + '.npy'
                X_data = np.load(self.disk_storage_path + "X_" + batch_file)
                y_data = np.load(self.disk_storage_path + "y_" + batch_file)
                return X_data, y_data
            elif self.storage == "disk_raw":
                batch_dir = self.disk_storage_path + "batch" + str(self.batch_id)
                for i in range(0, len(self.y_data)):
                    raise NotImplementedError
        else:
            pass

    def storeBatch(self, storage, X_data, y_data):
        """ Function to store a batch in memory / disk / or elsewhere """
        # data to store
        data_dict = {'X_data': X_data,
                     'y_data': y_data}
        # store in memory
        if storage == "memory":
            # dict with data
            self.raw_data = data_dict
        # store on disk
        elif storage == "disk":
            # check batch directory
            batch_file = 'batch' + str(self.batch_id) + '.pkl'
            if batch_file in os.listdir(self.disk_storage_path):
                print("Overwriting: %s" % batch_file)
            else:
                pkl.dump(data_dict,
                         open(self.disk_storage_path + batch_file, "wb"))
        # save in native numpy format
        elif storage == "numpy":
            batch_file = 'batch' + str(self.batch_id) + '.npy'
            if batch_file in os.listdir(self.disk_storage_path):
                print("Overwriting: %s" % batch_file)
            else:
                np.save(self.disk_storage_path + "X_" + batch_file, X_data)
                np.save(self.disk_storage_path + "y_" + batch_file, y_data)
        # save to disk as images
        elif storage == "disk_raw":
            # create new directory for batch
            batch_dir = self.disk_storage_path + "batch" + str(self.batch_id)
            os.mkdir(batch_dir)
            # loop through all images and store
            for i in range(0, X_data.shape[0]):
                img = array_to_img(X_data[i, :, :, :])
                img.save(batch_dir + "/" + i + ".jpeg")
        else:
            # do nothing
            return None

        # set flags and parameters
        self.is_stored = True
        self.storage = storage


class DataFetcher(object):
    """ Class to generate large batches of training data to
        generate mini-batches from for training
    - ideally use as few batches as possible
    - batches are kept in memory if possible, else stored on disk

    Arguments:
        data_dict:      ImageDir, full data set to generate batches from
        batch_size:     int, batch size in number of observations
        n_big_batches:  int, number of batches to generate
        image_size:     tupel (x,y,depth), image format
        random_shuffle_batches: boolean, randomly shuffle before
            generating batches
        asynch_read:    boolean, read urls from data_dict in asynchronous way
        disk_scratch:   str, path to save temporary files to
    -----------
    """
    def __init__(self, data_dict,
                 batch_size=None,
                 n_big_batches=None,
                 image_size=None,
                 random_shuffle_batches=False,
                 asynch_read=True,
                 disk_scratch=None):
        """ initialize """
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.n_big_batches = n_big_batches
        self.asynch_read = asynch_read
        self.n_observations = len(data_dict.paths)
        self.current_batch_id = 0
        self.random_shuffle_batches = random_shuffle_batches
        self.batches = None
        # resize images to specified size
        self.image_size = image_size
        # number of generated batches
        self.n_batches = None

        # batch data
        self.batches_stored = dict()

        # disk scratch to store temporary data
        self.disk_scratch = disk_scratch

        # define batches
        self._defineBatches()

        # Instantiate Image Loader
        self.imageLoader = ImageUrlLoader(parallel=asynch_read)

        # check input
        assert (n_big_batches is None) | (batch_size is None)
        assert (n_big_batches is not None) | (batch_size is not None)

    def _defineBatches(self):
        """ Function to split observations into batches """
        # extract all ids
        all_keys = list(self.data_dict.unique_ids)

        # randomly shuffle keys
        if self.random_shuffle_batches:
            random.shuffle(all_keys)

        # create batches based on number of batches
        if self.n_big_batches is not None:
            self.n_big_batches += 1
            # define cuts for batches
            cuts = np.linspace(0, self.n_observations,
                               self.n_big_batches).round()
        # create batches based on batch size
        elif self.batch_size is not None:
            cuts = [x for x in range(0, self.n_observations,
                                     int(self.batch_size))]
            if cuts[-1] < self.n_observations:
                cuts.append(self.n_observations)

        # convert batch sizes to integers
        cuts = [int(x) for x in cuts]

        # save batches into dictionary
        batches = dict()
        for i in range(0, (len(cuts) - 1)):
            # create DataBatch object
            current_batch = DataBatch(ids=all_keys[cuts[i]:cuts[i+1]],
                                      batch_id=i)
            current_batch.setDiskStoragePath(self.disk_scratch)
            batches[i] = current_batch

        # save batches
        self.n_batches = len(batches.keys())
        self.batches = batches

    def _listOfImagesToNumpy(self, images):
        """ Converts list of image objects to numpy array """
        # build image data array, y_labels
        for i in range(0, len(images)):
            if self.image_size is not None:
                img = images[i].resize(self.image_size)
            else:
                img = images[i]
            img_arr = img_to_array(img)
            if i == 0:
                dims = [len(images)] + list(img_arr.shape)
                X_data = np.zeros(shape=dims)
            X_data[i, :, :, :] = img_arr

        return X_data

    def nextBatch(self, batch_to_get_id=None):
        """ Get next batch or specify specific batch id to get """

        # batch id to get
        if batch_to_get_id is None:
            batch_to_get_id = self.current_batch_id

        # batch to get
        batch_to_get = self.batches[batch_to_get_id]

        # check if batch is available in memory / disk
        if batch_to_get.is_stored:
            # get batch data
            X_data, y_data = batch_to_get.getBatchData()
            # return X np array, label array
            return X_data, y_data

        # get data of current batch
        urls = list()

        for key in batch_to_get.ids:
            value = self.data_dict.data_dict[key]
            batch_to_get.batch_subjects[key] = value
            batch_to_get.y_data.append(value['label'])
            urls.append(value['path'])

        # get images using Image Loader class
        binary_images = self.imageLoader.getImages(urls)

        # convert images to array
        X_data = self._listOfImagesToNumpy(images=binary_images)
        y_data = np.array(batch_to_get.y_data)

        # decide where to store batch
        system_memory_usage_percent = psutil.virtual_memory()[2]
        if (system_memory_usage_percent < 90):
            save_to = "memory"
        elif self.disk_scratch is not None:
            save_to = "disk"
        elif self.disk_scratch is not None:
            save_to = "disk_raw"
        else:
            save_to = "none"

        # store batch
        batch_to_get.storeBatch(storage=save_to, X_data=X_data,
                                y_data=y_data)

        # increment current batch
        if self.current_batch_id < (self.n_batches-1):
            self.current_batch_id += 1
        else:
            self.current_batch_id = 0

        # return X np array, label array
        return X_data, y_data

    def storeAllOnDisk(self, path):
        """ store all images on disk in class specific folders """
        # fetch meta data
        urls = list()
        y_data = self.data_dict.labels
        ids = self.data_dict.unique_ids
        urls = self.data_dict.paths

        # save in chunks of 1000 images
        cuts = [x for x in range(0, self.n_observations, 1000)]
        if cuts[-1] < self.n_observations:
            cuts.append(self.n_observations)

        # convert batch sizes to integers
        cuts = [int(x) for x in cuts]

        for i in range(0, (len(cuts) - 1)):

            idx = [x for x in range(cuts[i], cuts[i+1])]

            current_ids = [ids[z] for z in idx]
            current_urls = [urls[z] for z in idx]
            current_y = [y_data[z] for z in idx]

            # invoke asynchronous read
            binary_images = self.imageLoader.getImages(current_urls)

            # store on disk
            img_id = 0
            for c_id, c_y in zip(current_ids, current_y):
                # check directory
                if not os.path.isdir(path + str(c_y)):
                    os.mkdir(path + str(c_y))
                # define path
                path_img = path + str(c_y) + "/" + \
                           str(c_id) + ".jpeg"
                img = binary_images[img_id]
                img = img.resize(self.image_size)
                img.save(path_img)
                img_id += 1
        return None


if __name__ == "__main__":
    import time
    gen = DataFetcher(test_dir, asynch_read=True, image_size = (28, 28),
                      batch_size=100,
                      disk_scratch = 'D:/Studium_GD/Zooniverse/Data/transfer_learning_project/scratch/')

    gen.storeAllOnDisk(path = 'D:/Studium_GD/Zooniverse/Data/transfer_learning_project/scratch/')
    time_start = time.time()
    X_data, y_data = gen.nextBatch()
    print("Required: %s seconds" % (time.time()-time_start))
    X_data2, y_data2 = gen.nextBatch(batch_to_get_id=0)
    print("Required: %s seconds" % (time.time()-time_start))
    url_test = 'https://panoptes-uploads.zooniverse.org/production/subject_location/a44e26b9-be8a-453e-9af0-5899c17e7efc.jpeg'
    response = requests.get(url_test)
    img = Image.open(BytesIO(response.content))
    assert(np.array_equal(X_data, X_data2))


    gen = DataFetcher(test_dir, asynch_read=True, image_size = (28, 28),
                      batch_size=100)
    time_start = time.time()
    X_data, y_data = gen.nextBatch()
    print("Required: %s seconds" % (time.time()-time_start))
    X_data2, y_data2 = gen.nextBatch(batch_to_get_id=0)
    print("Required: %s seconds" % (time.time()-time_start))
    np.array_equal(X_data, X_data2)
    assert(np.array_equal(X_data, X_data2))





