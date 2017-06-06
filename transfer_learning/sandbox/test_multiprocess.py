# test multiprocessing of data loader
from main import prep_data

train_dir, test_dir, val_dir = prep_data()




    # test
    img_loader = ImageUrlLoader()
    # get some image urls
    urls = train_dir.paths[512:630]

    time_start = time.time()
    imgs = img_loader.getImages(urls)
    time_end = time.time()