# import modules
from PIL import Image
import requests
from io import BytesIO
import aiohttp
import asyncio
import async_timeout
import time
import os


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

    def _getAsyncUrls(self, urls, ids):
        """ Load multiple urls in a parallel way """

        # prepare result dictionary
        images_dict = dict()

        # define asynchronous functions
        async def download_coroutine(session, key, url):
            # with async_timeout.timeout(180):
            async with session.get(url) as response:
                while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    try:
                        img = Image.open(BytesIO(chunk))
                        images_dict[key] = img
                    except IOError:
                        print("Could not access image: %s \n" % url)
            return await response.release()

        # asynchronous main loop
        async def main(loop):
            async with aiohttp.ClientSession(loop=loop) as session:
                tasks = [download_coroutine(session, key, url) for
                         key, url in zip(ids, urls)]
                await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(loop))
        return images_dict

    def _getAsyncUrls2(self, urls, ids):
        """ Load multiple urls in a parallel way """

        # prepare result dictionary
        images_dict = dict()

        # define asynchronous functions
        async def download_coroutine(session, key, url):
            # with async_timeout.timeout(180):
            async with session.get(url) as response:
                while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    try:
                        img = Image.open(BytesIO(chunk))
                        images_dict[key] = img
                    except IOError:
                        print("Could not access image: %s \n" % url)
            return await response.release()

        # asynchronous main loop
        async def main(loop):
            async with aiohttp.ClientSession(loop=loop) as session:
                tasks = [download_coroutine(session, key, url) for
                         key, url in zip(ids, urls)]
                await asyncio.gather(*tasks)

        # crate new event loop to work in multithreaded environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(loop))
        return images_dict

    def getImages(self, urls):
        """ Retrieve images from list of urls """

        # number of urls
        size = len(urls)

        # prepare result list
        res = list()

        # sequential retrieval
        if any((not self.parallel, size == 1)):
            # loop through urls and return
            for url in urls:
                img = self._getOneImageFromURL(url)
                res.append(img)
        # invoke asynchronous read
        else:
            # create dummy ids
            internal_ids = [x for x in range(0, size)]

            # invoke parallel read
            res_dict = self._getAsyncUrls2(urls, internal_ids)

            # ensure correct ordering
            for i in internal_ids:
                res.append(res_dict[i])

        # return list of image objects
        return res

    def storeOnDisk(self, urls, labels, ids, path, target_size=None,
                    chunk_size=1000, overwrite=False, create_path=True):
        """ store all images on disk in class specific folders """

        # check
        if not os.path.exists(path) & create_path:
            os.mkdir(path)
        else:
            NameError("Path not Found")

        # ensure / at end of path
        if path[-1] != '/':
            path = path + '/'

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
                f = [n.split(".")[0] for n in os.listdir(path + str(sub_dir))]
                existing_files.extend(f)

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

        if size == 0:
            print("Everything already on disk")
            return None

        # define chunks of images to load
        cuts = [x for x in range(0, size, chunk_size)]
        if cuts[-1] < size:
            cuts.append(size)

        # convert chunk sizes to integers
        cuts = [int(x) for x in cuts]

        # initialize progress counter
        jj = 0

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
            binary_images = self.getImages(chunk_urls)

            # store on disk
            img_id = 0
            for c_id, c_y in zip(chunk_ids, chunk_y):
                # define path
                path_img = path + str(c_y) + "/" + \
                           str(c_id) + ".jpeg"

                # check if exists
                if os.path.exists(path_img):
                    continue
                # get current image
                img = binary_images[img_id]

                # resize if specified
                if target_size is not None:
                    img = img.resize(target_size)

                # save to disk
                img.save(path_img)
                img_id += 1

                # print progress
                jj += 1
                if jj % 500 == 0:
                    print("%s stored on disk" % (round(jj/size, 0)))

        return None

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
