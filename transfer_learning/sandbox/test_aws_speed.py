# Test AWS I/O performance

# load modules
from config.config import config
from tools.image_url_loader import ImageUrlLoader

from main import prep_data
import time
import random
import numpy as np

import logging as log


############################
# Read from URL Parallel
############################

def test_url_parallel(batch_sizes):
    # initialize URL Loader
    loader = ImageUrlLoader()

    # iterate through images in different batch sizes

    log.info("Read Parallel")
    for b in batch_sizes:
        # do 10 tests
        times = list()
        for i in range(0,10):
            # randomly sample urls
            samp = random.sample(train_dir.paths,b)
            # fetch urls
            time_st = time.time()
            imgs = loader.getImages(samp)
            time_e = time.time()
            time_req = time_e - time_st
            times.append(time_req)
        # print summary
        avg = np.average(times)
        std = np.std(times)
        per_sec = b / avg
        log.info("Batch Size: " + str(b))
        log.info("Average: %s, STD: %s, Image/s: %s" % (avg, std, per_sec))
        log.info("---------------------------------------------------------------")


############################
# Read from URL Sequential
############################

def test_url_seq(batch_sizes):

    # initialize URL Loader
    loader = ImageUrlLoader(parallel=False)


    # iterate through images in different batch sizes

    log.info("Read Sequential")
    for b in batch_sizes:
        # do 10 tests
        times = list()
        for i in range(0,10):
            # randomly sample urls
            samp = random.sample(train_dir.paths,b)
            # fetch urls
            time_st = time.time()
            imgs = loader.getImages(samp)
            time_e = time.time()
            time_req = time_e - time_st
            times.append(time_req)
        # print summary
        avg = np.average(times)
        std = np.std(times)
        per_sec = b / avg
        log.info("Batch Size: " + str(b))
        log.info("Average: %s, STD: %s, Image/s: %s" % (avg, std, per_sec))
        log.info("---------------------------------------------------------------")



###################
# Read from URL & save to disk
###################

def test_url_disk(batch_sizes):

    # initialize URL Loader
    loader = ImageUrlLoader(parallel=True)

    # iterate through images in different batch sizes
    project_id = config['projects']['panoptes_id']
    image_size_save = config[project_id]['image_size_save'].split(',')
    image_size_save = tuple([int(x) for x in image_size_save])

    log.info("Store On Disk")
    save_dir = config['paths']['path_scratch'] + 'aws_test'
    for b in batch_sizes:
        # do 10 tests
        times = list()
        for i in range(0, 10):
            # randomly sample urls
            samp_id = random.sample(train_dir.unique_ids, b)
            samp_url = [train_dir.info_dict[s]['url'] for s in samp_id]
            samp_labels = [train_dir.info_dict[s]['y_data'] for s in samp_id]

            # fetch urls
            time_st = time.time()
            loader.storeOnDisk(samp_url, samp_labels, samp_id, save_dir,
                               target_size=image_size_save[0:2])
            time_e = time.time()
            time_req = time_e - time_st
            times.append(time_req)
        # print summary
        avg = np.average(times)
        std = np.std(times)
        per_sec = b / avg
        log.info("Batch Size: " + str(b))
        log.info("Average: %s, STD: %s, Image/s: %s" % (avg, std, per_sec))
        log.info("---------------------------------------------------------------")

###################
# Read from Disk
###################

def test_read_disk(batch_sizes):
    # image directory
    img_dir = config['paths']['path_scratch'] + 'aws_test'
    project_id = config['projects']['panoptes_id']

    image_size_model = config[project_id]['image_size_model'].split(',')
    image_size_model = tuple([int(x) for x in image_size_model])

    # iterate through images in different batch sizes
    # batch_size = eval(config[project_id]['batch_size'])
    from tools.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1./255)

    log.info("Store On Disk")
    for b in batch_sizes:

        # use keras to perform tests
        generator = datagen.flow_from_directory(
            img_dir,
            target_size=image_size_model[0:2],
            batch_size=b,
            class_mode='binary')

        # do 10 tests
        times = list()
        for i in range(0, 10):

            # fetch urls
            time_st = time.time()
            X_data, y_data = generator.next()
            time_e = time.time()
            time_req = time_e - time_st
            times.append(time_req)
        # print summary
        avg = np.average(times)
        std = np.std(times)
        per_sec = b / avg
        log.info("Batch Size: " + str(b))
        log.info("Average: %s, STD: %s, Image/s: %s" % (avg, std, per_sec))
        log.info("---------------------------------------------------------------")


###############################
# Read from URL Keras Iterator
###############################

def test_read_url_keras(batch_sizes):
    # image directory
    img_dir = config['paths']['path_scratch'] + 'aws_test'
    project_id = config['projects']['panoptes_id']

    image_size_model = config[project_id]['image_size_model'].split(',')
    image_size_model = tuple([int(x) for x in image_size_model])

    # randomly sample urls
    urls = train_dir.paths
    labels = train_dir.labels

    # iterate through images in different batch sizes
    # batch_size = eval(config[project_id]['batch_size'])
    from tools.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1./255)

    log.info("URLs from Keras Iterator")
    for b in batch_sizes:

        # use keras to perform tests
        generator = datagen.flow_from_urls(
            urls=urls,
            labels=labels,
            classes=['0','1'],
            target_size=image_size_model[0:2],
            batch_size=b,
            class_mode='binary')

        # do 10 tests
        times = list()
        for i in range(0, 10):

            # fetch urls
            time_st = time.time()
            X_data, y_data = generator.next()
            time_e = time.time()
            time_req = time_e - time_st
            times.append(time_req)
        # print summary
        avg = np.average(times)
        std = np.std(times)
        per_sec = b / avg
        log.info("Batch Size: " + str(b))
        log.info("Average: %s, STD: %s, Image/s: %s" % (avg, std, per_sec))
        log.info("---------------------------------------------------------------")


if __name__ == '__main__':

    # log
    log.basicConfig(filename=config['paths']['path_scratch'] + '/aws_test.log',
                        level=log.DEBUG,
                        format='%(message)s')
    console = log.StreamHandler()
    log.getLogger('').addHandler(console)
    log.getLogger("requests").setLevel(log.WARNING)
    log.getLogger("urllib3").setLevel(log.WARNING)
    log.getLogger("asyncio").setLevel(log.WARNING)

    # get data
    train_dir, test_dir, val_dir = prep_data()

    # Params
    batch_sizes = [1, 10, 50, 100, 200, 1000]

    # run tests
    test_url_parallel(batch_sizes)
    test_url_seq([1, 10, 50, 100])
    test_url_disk(batch_sizes)
    test_read_disk(batch_sizes)
    test_read_url_keras([10, 50, 100])

    console.close()
