# function to fetch SS URLs via Ouroboros API
import aiohttp
import asyncio
from contextlib import closing


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


def get_oroboros_api_data(ids, quality='standard'):
    api_path = 'https://api.zooniverse.org/projects/serengeti/subjects/'
    url_list = list()
    for ii in ids:
        url_list.append(api_path + ii)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pages = get_multiple_pages(url_list, ids)
    all_urls = extract_data(pages, quality)
    return all_urls


if __name__ == '__main__':
    subject_ids = ['ASG001gpfu', 'ASG001k4o9', 'ASG001isc1', 'ASG001hw4x']
    data = get_oroboros_api_data(subject_ids)
    for d in data:
        print(d)

#








