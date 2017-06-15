def get_async_url_bytes2(self, current_batch):
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
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    try:
                        img = Image.open(BytesIO(chunk))

                        binary_images_dict[key] = img
                    except:
                        resp = requests.get(value['url'])
                        img = Image.open(BytesIO(resp.content))
                        binary_images_dict[key] = img
                        resp.close()
            return await response.release()

    # asynchronous main loop
    async def main(loop):
        async with aiohttp.ClientSession(loop=loop) as session:
            tasks = [download_coroutine(session, key, value) for
                     key, value in current_batch.items()]
            await asyncio.gather(*tasks)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    return binary_images_dict







def dummy():
    return 1,2,3


TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE,\
TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT,\
DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD = dummy()