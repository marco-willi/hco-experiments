"""
Class that implements functions to load images from urls
- uses asyncio to load images asyncronously
- exception handling regarding non-retrievable urls
"""

# import modules
from PIL import Image, ImageFile
import requests
from io import BytesIO
import aiohttp
import asyncio
import time
import os
from config.config import logging
from tools.helpers import second_to_str


# set option to avoid errors while opening images with PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Class to get images from URLs
# Returns Image objects or stores them on disk
# specify either parallel or sequential reading
class ImageUrlLoader(object):
    """ Load images from URLs """
    def __init__(self, parallel=True):
        self.parallel = parallel
        self.size = None

    def _getOneImageFromURL(self, url):
        """ Load one Image From URL """
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def _getAsyncUrls4(self, urls, ids):
        """ Load multiple urls in a parallel way, catch failed attempts """

        # prepare result dictionary
        images_dict = dict()

        # prepare list for fails
        failures = {'urls': list(), 'ids': list()}

        # define asynchronous functions
        async def download_coroutine(semaphore, session, key, url):
            # with async_timeout.timeout(180):
            async with semaphore:
                async with session.get(url) as response:
                    while True:
                        chunk = await response.content.read()
                        if not chunk:
                            break
                        try:
                            img = Image.open(BytesIO(chunk))
                            images_dict[key] = img
                        except:
                            logging.warn("Could not access image: %s w id %s"
                                         % (url, key))
                            success = False
                            counter = 0
                            n_attempts = 0
                            while ((not success) and (counter < n_attempts)):
                                logging.debug("Trying again")
                                time.sleep(0.1)
                                try:
                                    chunk = await response.content.read()
                                    img = Image.open(BytesIO(chunk))
                                    images_dict[key] = img
                                    success = True
                                except:
                                    counter += 1
                                    logging.warn("Failed Attempt %s / %s"
                                                 % (counter, n_attempts))
                            # add to failures list
                            if not success:
                                failures['urls'].append(url)
                                failures['ids'].append(key)
                                # log failures
                                for u, i in zip(failures['urls'],
                                                failures['ids']):
                                    logging.warn("Failed to access id: %s on\
                                                 url: %s" % (i, u))

                return await response.release()

        # asynchronous main loop
        async def main(loop):
            async with aiohttp.ClientSession(loop=loop) as session:

                # new: instantiate a semaphore before calling our coroutines
                semaphore = asyncio.BoundedSemaphore(1000)

                tasks = [download_coroutine(semaphore, session, key, url) for
                         key, url in zip(ids, urls)]
                await asyncio.gather(*tasks)

        # crate new event loop to work in multithreaded environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(loop))

        return images_dict, failures

    def getImages(self, urls, ids, zooniverse_imgproc=False, target_size=None):
        """ Retrieve images from list of urls """

        # number of urls
        size = len(urls)

        # prepare result list
        res = dict()

        # if to use the zooniverse imgproc server to pre-process images
        if (zooniverse_imgproc) & (target_size is not None):
            imgproc_url = (
                "https://imgproc.zooniverse.org/resize?o=!&w=%s&h=%s&u=" %
                (target_size[0], target_size[1]))

            for i in range(0, size):
                urls[i] = imgproc_url + urls[i].replace('https://', '')

        # sequential retrieval
        if any((not self.parallel, size == 1)):
            # loop through urls and return
            for url, i in zip(urls, ids):
                img = self._getOneImageFromURL(url)
                res[i] = img
        # invoke asynchronous read
        else:
            # invoke parallel read
            res, failures = self._getAsyncUrls4(urls, ids)

            # try to read failed images
            counter = 0
            while (len(failures['urls']) > 0) and counter < 10:
                res2, failures = self._getAsyncUrls4(urls, ids)

                # combine results
                if len(res.keys()) > 0:
                    res = {**res, **res2}

                # increase counter
                counter += 1
        return res

    def storeOnDisk(self, urls, labels, fnames, path, target_size=None,
                    chunk_size=1000, overwrite=False, create_path=True,
                    zooniverse_imgproc=False
                    ):
        """ store all images on disk in class specific folders
            urls: list of urls
            labels: list of folder-names in which image is going to be stored
            fnames: list of file names of images
            path: parent folder where to store image label folders and images

        """

        # filenames as identifiers
        ids = fnames

        # number of urls
        size_total = len(urls)

        # check
        if not os.path.exists(path) & create_path:
            os.mkdir(path)
        else:
            NameError("Path not Found")

        # ensure / at end of path
        if path[-1] != os.path.sep:
            path = path + os.path.sep

        # create dictionary for convenience
        data_dict = dict()
        for url, label, ii in zip(urls, labels, ids):
            data_dict[str(ii)] = [url, label]

        # create class specific directories
        existing_files = list()
        for sub_dir in set(labels):
            if not os.path.exists(path + str(sub_dir)):
                os.mkdir(path + str(sub_dir))
            # list all files
            else:
                logging.info("Listing Files of %s" % path + str(sub_dir))
                f = os.listdir(path + str(sub_dir))
                existing_files.extend(f)
                logging.info("Fnished Listing %s" % path + str(sub_dir))

        # remove already existing files from re-storing operation
        existing_files = set(existing_files)
        dict_ids = list(data_dict.keys())
        if not overwrite:
            if len(existing_files) > 0:
                for k in dict_ids:
                    if k in existing_files:
                        data_dict.pop(k)

        # get relevant data from dictionary
        dict_ids = list(data_dict.keys())
        size = len(dict_ids)

        # store info for return object
        summary = dict()
        failures_savings = dict()
        summary['failures'] = failures_savings
        summary['path'] = path

        # return if nothing needs to be done
        if size == 0:
            logging.info("Everything already on disk")
            return summary

        if size > 0:
            logging.info("Already on disk: %s" % (size_total - size))

        # define chunks of images to load
        cuts = [x for x in range(0, size, chunk_size)]
        if cuts[-1] < size:
            cuts.append(size)

        # convert chunk sizes to integers
        cuts = [int(x) for x in cuts]

        # initialize progress counter
        jj = 0
        time_begin = time.time()
        time_b = time.time()

        for i in range(0, (len(cuts) - 1)):

            idx = [x for x in range(cuts[i], cuts[i+1])]

            chunk_ids = [dict_ids[z] for z in idx]
            chunk_urls = list()
            chunk_y = list()
            for ci in chunk_ids:
                val = data_dict[ci]
                chunk_urls.append(val[0])
                chunk_y.append(val[1])

            # invoke asynchronous read
            binary_images = self.getImages(
                chunk_urls, chunk_ids,
                zooniverse_imgproc=zooniverse_imgproc,
                target_size=target_size)

            # store on disk
            img_id = 0

            for c_id, c_y in zip(chunk_ids, chunk_y):

                # define path
                path_img = path + str(c_y) + os.path.sep + \
                           str(c_id)

                # check if exists
                if os.path.exists(path_img):
                    continue
                # get current image
                try:
                    img = binary_images[c_id]
                except:
                    logging.warn("Could not access image %s - skipping..."
                                 % c_id)
                    failures_savings[c_id] = img_id
                    continue

                # resize if specified
                if target_size is not None:
                    try:
                        img = img.resize(target_size)
                    except:
                        logging.warn("Could not resize image %s - skipping..."
                                     % c_id)
                        failures_savings[c_id] = img_id
                        continue

                # save to disk
                try:
                    img.save(path_img)
                except:
                    logging.warn("Could not save image %s - skipping..."
                                 % c_id)
                    failures_savings[c_id] = img_id
                    continue

                img_id += 1

                # count processed images
                jj += 1

                # print progress
                if (jj % 500) == 0:
                    tm_now = time.time()
                    tm = round(tm_now - time_b, 0)
                    tm_total = second_to_str(time.time() - time_begin)
                    time_b = time.time()
                    logging.info("%s / %s stored on disk\
                                  , took %s s (Total: %s)"
                                 % (jj+size_total-size, size_total,
                                     tm, tm_total))

        return summary


if __name__ == "__main__":
    from main import prep_data
    train_dir, test_dir, val_dir = prep_data()

    # test
    img_loader = ImageUrlLoader()
    # get some image urls
    urls = train_dir.paths[512:630]

    time_start = time.time()
    imgs = img_loader.getImages(urls)
    time_end = time.time()
    print("To Fetch %s images in parallel, it took: %s seconds" %
          (len(imgs), time_end - time_start))

    # get some image urls
    time_start = time.time()
    img_loader2 = ImageUrlLoader(parallel=False)

    imgs2 = img_loader2.getImages(urls)
    time_end = time.time()
    print("To Fetch %s images in seq., it took: %s seconds" %
          (len(imgs), time_end - time_start))

    assert(imgs == imgs2)
