# import modules
import random
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import requests
from io import BytesIO
import aiohttp
import asyncio
import async_timeout
import time

class DataFetcher(object):
    """ Class to generate batches of training data """
    def __init__(self, data_dict,
                 batch_size=None,
                 n_big_batches=None,
                 image_size=None,
                 random_shuffle_batches=False,
                 asynch_read=True):
        """ initialize """
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.n_big_batches = n_big_batches
        self.asynch_read = asynch_read
        self.n_observations = len(data_dict.keys())
        self.current_batch = 0
        self.random_shuffle_batches = random_shuffle_batches
        self.batches = None
        self.image_size = image_size
        self.n_batches = None

        # define batches
        self.define_batches()

        # check input
        assert (n_big_batches is None) | (batch_size is None)
        assert (n_big_batches is not None) | (batch_size is not None)

    def define_batches(self):
        all_keys = list(self.data_dict.keys())
        # randomly shuffle batches
        if self.random_shuffle_batches:
            random.shuffle(all_keys)

        # create batches based on number of batches
        if self.n_big_batches is not None:
            self.n_big_batches += 1
            # define cuts for batches
            cuts = np.linspace(0, self.n_observations,
                               self.n_big_batches).round()

        # create batches based on batch size
        if self.batch_size is not None:
            cuts = [x for x in range(0, self.n_observations,
                                     int(self.batch_size))]
            if cuts[-1] < self.n_observations:
                cuts.append(self.n_observations)

        # ensure integers in batch sizes
        cuts = [int(x) for x in cuts]

        # define batches
        batches = dict()
        for i in range(0, (len(cuts) - 1)):
            current_batch = all_keys[cuts[i]:cuts[i+1]]
            batches[i] = current_batch

        # save batches
        self.n_batches = len(batches.keys())
        self.batches = batches


    def get_async_url_bytes(self, batch):
        """ Function to read data from a list of urls in an asynchronous way to
            speed up the retrieveal
        """
        # prepare result dictionary
        binary_images_dict = dict()

        # define asynchronous functions
        async def download_coroutine(session, key, value):
            with async_timeout.timeout(180):
                async with session.get(value['url']) as response:
                    while True:
                        chunk = await response.content.read()
                        if not chunk:
                            break
                        img = Image.open(BytesIO(chunk))
                        binary_images_dict[key] = img

                return await response.release()

        # asynchronous main loop
        async def main(loop):
            async with aiohttp.ClientSession(loop=loop) as session:
                tasks = [download_coroutine(session, key, value) for
                         key, value in batch.items()]
                await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(loop))
        return binary_images_dict

    def get_batch(self):
        """ Get next batch or specify specific batch """

        # get batch ids
        batch_to_get_id = self.batches[self.current_batch]

        # get data of current batch
        batch_to_get = dict()
        for key, value in self.data_dict.items():
            if key in batch_to_get_id:
                batch_to_get[key] = value

        # invoke asynchronous read
        if self.asynch_read is True:
            binary_images = self.get_async_url_bytes(batch_to_get)
        # invoke sequential read
        else:
            binary_images = dict()
            for key, value in batch_to_get.items():
                response = requests.get(value['url'])
                img = Image.open(BytesIO(response.content))
                binary_images[key] = img

        # build image data array, y_labels and original subject ids
        y_data = list()
        original_ids = list()
        i = 0
        for key, value in batch_to_get.items():
            if self.image_size is not None:
                img = binary_images[key].resize(self.image_size)
            else:
                img = binary_images[key]
            img_arr = img_to_array(img)
            if i == 0:
                dims = [len(batch_to_get.keys())] + list(img_arr.shape)
                X_data = np.zeros(shape=dims)
            X_data[i, :, :, :] = img_arr
            i += 1
            y_data.append(int(value['y_label']))
            original_ids.append(value['subject_id'])

        y_data = np.array(y_data)

        # increment current batch
        if self.current_batch < (self.n_batches-1):
            self.current_batch += 1
        else:
            self.current_batch = 0

        # return X np array, label array, original ids
        return X_data, y_data, original_ids


if __name__ == "__main__":
    gen = DataFetcher(test_dict, asynch_read=True, image_size = (28, 28),
                      batch_size=100)
    time_start = time.time()
    X_data, y_data, original_ids = gen.get_batch()
    print("Required: %s seconds" % (time.time()-time_start))
    X_data2, y_data2, original_ids2 = gen.get_batch()
    print("Required: %s seconds" % (time.time()-time_start))
    url_test = 'https://panoptes-uploads.zooniverse.org/production/subject_location/a44e26b9-be8a-453e-9af0-5899c17e7efc.jpeg'
    response = requests.get(url_test)
    img = Image.open(BytesIO(response.content))
