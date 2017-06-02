# import modules
from PIL import Image
import requests
from io import BytesIO

class ImageDir(object):
    """ Class to save image directory """
    def __init__(self, paths, labels, unique_ids, info_dict):
        """
            paths: list with paths/urls to images
            labels: list with labels
            unique_ids: unique_ids
            info_dict: dictionary with additional info, key: unique_ids
        """
        self.paths = paths
        self.labels = labels
        self.unique_ids = unique_ids
        self.info_dict = info_dict

        # build dictionary with all information
        self.data_dict = dict()
        for u_id, i in zip(unique_ids, [x for x in range(0, len(unique_ids))]):
            self.data_dict[u_id] = {'path': paths[i],
                                    'label': labels[i],
                                    'info_dict': info_dict}

    def getImageInfo(self, key):
        """ Returns Image Directory Info """
        return self.info_dict[key]

    def getOneImage(self, key):
        """ Returns Image Object """
        img_dir = self.data_dict[key]
        response = requests.get(img_dir['path'])
        img = Image.open(BytesIO(response.content))
        return img