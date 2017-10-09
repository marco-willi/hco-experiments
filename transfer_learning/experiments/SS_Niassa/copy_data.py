""" Copy & Resize Image Data """
import os
import re
from PIL import Image as pil_image
from collections import OrderedDict


# Parameters
path_root = "/home/packerc/shared/albums/MountainZebra/MZNP_S4/"
#path_root = "/home/packerc/will5448/data/Niassa/Niassa_S1"
path_target = "/home/packerc/will5448/data/MountainZebra_S4"

# size of the images
target_size = (500, 500)

# Functions
def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def list_files(path, ext='jpg|jpeg|bmp|png|ppm'):
    """ Returns dict of paths of all files in the path, including sub-dirs
        key: filename, value: full path
    """
    file_paths = OrderedDict()
    for path, subdirs, files in os.walk(path):
        for name in files:
            if re.match(r'([\w]+\.(?:' + ext + '))', name):
                path_full = os.path.join(path, name)
                file_paths[name] = path_full
    return file_paths


if __name__ == "__main__":
    # get all image paths (and remove already processed files)
    image_files_root = list_files(path_root)
    n_files = len(image_files_root.keys())
    image_files_target = list_files(path_target)
    for ii in image_files_target.keys():
        if ii in image_files_root:
            image_files_root.pop(ii, None)

    # load, resize and save images
    counter=0
    for img_name, img_path in image_files_root.items():
        try:
            img = load_img(img_path, target_size=(500, 500))
        except:
            print("Load of img %s failed" % img_path)
            continue
        img.save(os.path.join(path_target, img_name))
        counter+=1
        if (counter % 1000) == 0:
            print("Processed %s / %s" % (counter, n_files))
