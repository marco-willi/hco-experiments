import aiohttp
import asyncio
import async_timeout
import os
import random
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import requests
from io import BytesIO
import time
from multiprocessing import Pool


#urls = ['https://images-na.ssl-images-amazon.com/images/G/01/img15/pet-products/small-tiles/30423_pets-products_january-site-flip_3-cathealth_short-tile_592x304._CB286975940_.jpg',
#        'http://www.rd.com/wp-content/uploads/sites/2/2016/04/01-cat-wants-to-tell-you-laptop.jpg',
#        'http://i.huffpost.com/gen/3152148/images/o-ANIMALS-FUNNY-facebook.jpg']


obs = list()
async def download_coroutine(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            chunk = response.read()
            obs.append(chunk)
            print(chunk)
            #img = await Image.open(BytesIO(chunk))
            return await chunk



async def main(loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, url) for url in urls[0:5]]
        await asyncio.gather(*tasks)


loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))



urls_all = urls


import aiohttp
import asyncio
import async_timeout
import os

def get_one_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


time_start = time.time()
time_req = 0
obs = list()
res_dict = dict()
async def download_coroutine(session, url):
    with async_timeout.timeout(50):
        async with session.get(url) as response:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
                img = Image.open(BytesIO(chunk))
                obs.append(img)
        return await response.release()

async def main(loop):

    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, url) for url in urls]
        await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))
time_end = time.time()
time_req = time_req + (time_end - time_start)
print("Req %f" % time_req)















#################################
# Test with dictionary
#################################



import aiohttp
import asyncio
import async_timeout
import os

time_start = time.time()
time_req = 0
obs = list()
res_dict = dict()

async def download_coroutine(session, key, value):
    with async_timeout.timeout(50):
        async with session.get(value['url']) as response:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
                img = Image.open(BytesIO(chunk))
                res_dict[key] = img
        return await response.release()

async def main(loop):

    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, key, value) for key, value in this_batch.items()]
        await asyncio.gather(*tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))
time_end = time.time()
time_req = time_req + (time_end - time_start)
print("Req %f" % time_req)




