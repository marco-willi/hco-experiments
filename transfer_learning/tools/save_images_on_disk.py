# module to save images on disk
from tools.model_helpers import model_param_loader
from tools.image_url_loader import ImageUrlLoader
import time


def save_on_disk(subject_set, config, cfg_path, set_name):
    ##################################
    # Parameters
    ##################################

    cfg = model_param_loader(config)

    print_separator = "---------------------------------"

    ##################################
    # Save data on disk
    ##################################

    # define loaders
    data_loader = ImageUrlLoader()

    # save to disk
    print(print_separator)
    print("Saving %s data ...." % set_name)
    time_s = time.time()
    urls, labels, ids = subject_set.getAllURLsLabelsIDs()
    data_loader.storeOnDisk(urls=urls,
                            labels=labels,
                            ids=ids,
                            path=cfg_path['images'] + set_name,
                            target_size=cfg['image_size_save'][0:2],
                            chunk_size=100)

    print("Finished saving on disk after %s minutes" %
          ((time.time() - time_s) // 60))
    print(print_separator)
