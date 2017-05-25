#!/usr/bin/env python3
import asyncio
import logging
from contextlib import closing
import aiohttp # $ pip install aiohttp
from sandbox.url2filename import *


@asyncio.coroutine
def get(*args, **kwargs):
    response = yield from aiohttp.request('GET', *args, **kwargs)
    return (yield from response.read_and_close(decode=True))


def first_magnet(page):
    img = Image.open(BytesIO(page.content))
    return img

@asyncio.coroutine
def print_magnet(query):
    url = 'http://thepiratebay.se/search/{}/0/7/0'.format(query)
    page = yield from get(url, compress=True)
    magnet = first_magnet(page)
    print('{}: {}'.format(query, magnet))



@asyncio.coroutine
def get_one_image(url, session, semaphore, chunk_size=1<<15):
    with (yield from semaphore): # limit number of concurrent downloads
        filename = url2filename(url)
        logging.info('downloading %s', filename)
        response = yield from session.get(url)

        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    return filename, img



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
with closing(asyncio.get_event_loop()) as loop, \
     closing(aiohttp.ClientSession()) as session:
    semaphore = asyncio.Semaphore(4)
    download_tasks = (get_one_image(url, session, semaphore) for url in urls)
    result = loop.run_until_complete(asyncio.gather(*download_tasks))



async def get_one_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            img = await Image.open(BytesIO(resp.content))
            images_all.append(img)


loop = asyncio.get_event_loop()
images_all = []
future = asyncio.ensure_future(get_one_image(urls))
loop.run_until_complete(future)



#!/usr/bin/env python3
from multiprocessing.dummy import Pool # use threads for I/O bound tasks
from urllib.request import urlretrieve
# test parallel-processing
def urlretrieve(url):
    response = requests.get(url)
    return response


result = Pool(4).map(urlretrieve, urls) # download 4 files at a time






#!/usr/local/bin/python3.5
import asyncio
import requests

async def main():
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://www.google.com')
    future2 = loop.run_in_executor(None, requests.get, 'http://www.google.co.uk')
    response1 = await future1
    response2 = await future2
    print(response1.text)
    print(response2.text)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())




def get_one_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def process_id_range(id_range, store=None):
    """process a number of ids, storing the results in a dict"""
    if store is None:
        store = {}
    for id in id_range:
        store[id] = get_one_image(id)
    return store

from threading import Thread

def threaded_process(nthreads, id_range):
    """process the id range in a specified number of threads"""
    store = {}
    threads = []
    # create the threads
    for i in range(nthreads):
        ids = id_range[i::nthreads]
        t = Thread(target=process_id_range, args=(ids,store))
        threads.append(t)

    # start the threads
    [ t.start() for t in threads ]
    # wait for the threads to finish
    [ t.join() for t in threads ]
    return store

id_range = urls
tic = time.time()
reference = process_id_range(id_range)
reftime = time.time() - tic
print(reftime)

nlist = [1,2,4,8]
tlist = [reftime]
for nthreads in nlist[1:]:
    tic = time.time()
    ans = threaded_process(nthreads, id_range)
    toc = time.time()
    print(str(nthreads) + " | " + str(toc-tic))
    assert ans == reference
    tlist.append(toc-tic)

tt = process_id_range(urls)






