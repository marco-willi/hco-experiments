import aiohttp
import asyncio
import async_timeout
import os


async def download_coroutine(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
        return chunk


async def main(loop):
    urls = ["http://www.snapshotserengeti.org/subjects/standard/51a2c7dfe18f49172b003b88_0.jpg",
        "http://www.irs.gov/pub/irs-pdf/f1040a.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040ez.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040es.pdf",
        "http://www.irs.gov/pub/irs-pdf/f1040sb.pdf"]

    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, url) for url in urls]
        await asyncio.gather(*tasks)

    return tasks


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(main(loop))