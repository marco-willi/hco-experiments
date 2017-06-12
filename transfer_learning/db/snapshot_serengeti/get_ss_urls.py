# function to fetch SS URLs via Ouroboros API
import aiohttp
import asyncio
from contextlib import closing
import time
from config.config import path_cfg


async def fetch_page(session, url):
    """ Get one page """
    # with aiohttp.Timeout(500):
    async with session.get(url) as response:
        return await response.json()


def get_multiple_pages(urls, ids):
    """ Get multiple pages """
    tasks = []
    pages = []
    with closing(asyncio.get_event_loop()) as loop:
        with aiohttp.ClientSession(loop=loop) as session:
            for url in urls:
                tasks.append(fetch_page(session, url))
            pages = loop.run_until_complete(asyncio.gather(*tasks))
    return pages


def extract_data(pages, quality):
    """ extract relevant info from pages """
    res = dict()
    for p in pages:
        res[p['zooniverse_id']] = p['location'][quality]
    return res


    api_path = 'https://api.zooniverse.org/projects/serengeti/subjects/'
    url_list = list()
    for ii in ids:
        url_list.append(api_path + ii)

    size = len(ids)

    # define chunks of images to load
    cuts = [x for x in range(0, size, batch_size)]
    if cuts[-1] < size:
        cuts.append(size)

    # convert chunk sizes to integers
    cuts = [int(x) for x in cuts]

    all_urls = dict()

    # loop over chunks
    for i in range(0, (len(cuts) - 1)):

        time_s = time.time()

        # chunk ids
        idx = [x for x in range(cuts[i], cuts[i+1])]

        # store chunk urls and ids
        chunk_urls = list()
        chunk_ids = list()
        for ii in idx:
            chunk_urls.append(url_list[ii])
            chunk_ids.append(ids[ii])

        # call asyncio loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pages = get_multiple_pages(chunk_urls, chunk_ids)

        chunk_res = extract_data(pages, quality)

        # save results
        all_urls = {**all_urls, **chunk_res}

        time_req = time.time() - time_s

        print("%s / %s urls processed" % (cuts[i+1], size))
        print("last batch required %s minutes" % (time_req // 60))

        if log:
            log_file = path_cfg['logs'] + 'oroboros_log.txt'
            file = open(log_file,'a')
            print("%s / %s urls processed" % (cuts[i+1], size), file=file)
            print("last batch required %s minutes" % (time_req // 60), file=file)
            file.close()



    return all_urls


if __name__ == '__main__':
    ids = ['ASG001gpfu', 'ASG001k4o9', 'ASG001isc1', 'ASG001hw4x']
    from db.snapshot_serengeti.get_dryad_data import get_dryad_ss_data
    dd = get_dryad_ss_data(retrieve=False)
    ids = list(dd.keys())[0:1003]
    data = get_oroboros_api_data(ids, batch_size=100)
#    for d in data:
#        print(d)

#








