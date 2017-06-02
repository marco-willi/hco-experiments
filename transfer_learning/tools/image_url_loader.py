# import modules
from PIL import Image
import requests
from io import BytesIO
import aiohttp
import asyncio
import async_timeout


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
            with async_timeout.timeout(180):
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
            res_dict = self._getAsyncUrls(urls, internal_ids)

            # ensure correct ordering
            for i in internal_ids:
                res.append(res_dict[i])

        # return list of image objects
        return res


if __name__ == "__main__":
    from main import train_dir
    # test
    img_loader = ImageUrlLoader()
    # get some image urls
    urls = [train_dir.getDict()[u]['url'] for u in
            list(train_dir.getIds())[0:10]]
    imgs = img_loader.getImages(urls)

    # get some image urls
    img_loader2 = ImageUrlLoader(parallel=False)
    imgs2 = img_loader2.getImages(urls)

    assert(imgs == imgs2)
