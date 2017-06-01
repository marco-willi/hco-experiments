# import modules
from PIL import Image
import requests
from io import BytesIO

class ImageDir(object):
    """ Class to save image directory """
    def __init__(self, image_dict):
        """
            image_dict: dictionary with id as key and as value a dictionary:
                y_label, url, original_id, any other information
        """
        self.image_dict = image_dict

    def getDict(self):
        return self.image_dict

    def getIds(self):
        return self.image_dict.keys()

    def getImageInfo(self, key):
        """ Returns Image Directory Info """
        return self.image_dict[key]

    def getOneImage(self, key):
        """ Returns Image Object """
        img_dir = self.getImageInfo(key)
        response = requests.get(img_dir['url'])
        img = Image.open(BytesIO(response.content))
        return img

