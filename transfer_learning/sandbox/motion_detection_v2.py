# Test Motion Detection
import numpy as np
import os
from keras.preprocessing.image import img_to_array,  load_img, array_to_img
from scipy import ndimage
from skimage.filters import threshold_isodata, threshold_otsu, threshold_local
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter)
import collections


def find_max_mass(img, window_shape, step_size=50, result="corners"):

    # calculate windows to check
    right_corners = [window_shape[1]]
    while right_corners[-1] < (img.shape[1]-1):
        right_corners.append(right_corners[-1] + step_size)
    right_corners[-1] = (img.shape[1]-1)
    upper_corners = [window_shape[0]]
    while upper_corners[-1] < (img.shape[0]-1):
        upper_corners.append(upper_corners[-1] + step_size)
    upper_corners[-1] = (img.shape[0]-1)

    # loop over all windows and calculate mass
    max_mass = None
    for right in right_corners:
        for upper in upper_corners:
            left = right - window_shape[1]
            lower = upper - window_shape[0]
            mass = np.sum(img[lower:upper, left:right], axis=(0, 1))
            if max_mass is None:
                max_mass = mass
            if mass >= max_mass:
                max_left = left
                max_right = right
                max_lower = lower
                max_upper = upper
                max_mass = mass
        # print("left: %s, right: %s, upper: %s lower: %s - Mass: %s" %
        #       (left, right, upper, lower, mass))
    if result == "corners":
        res = {'upper': int(max_upper),
               'left': int(max_left),
               'lower': int(max_lower),
               'right': int(max_right)}

        # return max_left, max_right, max_lower, max_upper, max_mass
        return res
    elif result == "center":
        return ((max_upper + max_lower) / 2, (max_left + max_right) / 2)


def rectangle_coordinates_from_center(center_x, center_y, height,
                                      width, max_shape=None):
    # calculate rectangle boundaries
    upper = center_y + int(np.floor((height / 2)))
    lower = center_y - int(np.floor((height / 2)))
    left = center_x - int(np.floor((width / 2)))
    right = center_x + int(np.floor((width / 2)))

    # ensure boundaries are within full image if max_shape is specified
    if max_shape is not None:

        # calculate extensions to fit max window
        remove_from_lower = abs(np.max([upper - max_shape[0], 0]))
        add_to_upper = abs(np.min([lower, 0]))
        remove_from_left = abs(np.max([right - max_shape[1], 0]))
        add_to_right = abs(np.min([left, 0]))

        # add / subtract extensions
        lower = lower - remove_from_lower
        upper = upper + add_to_upper
        left = left - remove_from_left
        right = right + add_to_right

        # ensure non-negative indices
        upper = np.min([upper, max_shape[0]-1])
        lower = np.max([lower, 0])
        left = np.max([left, 0])
        right = np.min([right, max_shape[1]-1])

    else:
        # ensure non-negative indices
        upper = np.max([upper, 0])
        lower = np.max([lower, 0])
        left = np.max([left, 0])
        right = np.max([right, 0])

    res = {'upper': int(upper),
           'left': int(left),
           'lower': int(lower),
           'right': int(right)}
    return res

def motion_share(img):
    """ Assesses how much fuzzy patterns are in a binary image
        - used to decide which algorithm to use for a specific image
    """

    # ensure 3 dims
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    # calculate row and column sums
    row_sums = np.sum(img, axis=(1, 2))
    col_sums = np.sum(img, axis=(0, 2))

    # calculate share with non-negative values
    p_row = np.sum(row_sums > 0)/row_sums.shape[0]
    p_col = np.sum(col_sums > 0)/col_sums.shape[0]

    # calculate mean fuzzyness

    return (p_row + p_col) / 2


# rectangle_coordinates_from_center(center_x=50, center_y=50, height=20, width=20, max_shape=(20, 30))

def rectangle_proposal(img, min_rectangle_size, keep_aspect_ratio=True):
    """ Function that proposes a rectangle from a binary image
        of which the purpose is to capture as much of the mass and as little
        of the non-mass as possible
    """
    # inspect column and row distribution
    # look for areas with zero colum/row sums
    # this indicates clear patterns in one region of the image
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    row_sums = np.sum(img, axis=(1, 2))
    col_sums = np.sum(img, axis=(0, 2))
    thresh_rows = threshold_otsu(row_sums)
    thresh_cols = threshold_otsu(col_sums)
    min_row_id = np.argmax(row_sums > thresh_rows)
    max_row_id = row_sums.shape[0] - 1 -\
        np.argmax(np.flip(row_sums, axis=0) > thresh_rows)
    min_col_id = np.argmax(col_sums > thresh_cols)
    max_col_id = col_sums.shape[0] - 1 -\
        np.argmax(np.flip(col_sums, axis=0) > thresh_cols)
    # calculate center and draw rectangle
    row_center = (min_row_id + max_row_id) / 2
    col_center = (min_col_id + max_col_id) / 2
    rectangle_height = np.max([min_rectangle_size[0], max_row_id - min_row_id])
    rectangle_width = np.max([min_rectangle_size[1], max_col_id - min_col_id])
    rectangle_coord = rectangle_coordinates_from_center(
        center_x=col_center, center_y=row_center,
        height=rectangle_height, width=rectangle_width,
        max_shape=img.shape[0:2])
    if keep_aspect_ratio:
        rectangle_coord = correct_for_aspect_ratio(
            rectangle_coords=rectangle_coord,
            target_shape=min_rectangle_size,
            max_shape=img.shape)

    return rectangle_coord


def correct_for_aspect_ratio(rectangle_coords, target_shape, max_shape):
    """ Correct rectangle coords for given target_shape """

    # calc original aspect ratio
    original_ratio = target_shape[0] / target_shape[1]

    # calc proposed ratio
    width = rectangle_coords['right'] - rectangle_coords['left']
    height = rectangle_coords['upper'] - rectangle_coords['lower']

    rect_ratio = height / width

    # correct
    if (original_ratio < (rect_ratio*1.1)) and\
       (original_ratio > (rect_ratio*0.9)):
        return rectangle_coords
    elif original_ratio < rect_ratio:
        # calc deviation
        correct_factor = rect_ratio / original_ratio
        # increase width
        width_delta = int(((width * correct_factor) - width) / 2)
        rectangle_coords['right'] += width_delta
        rectangle_coords['left'] -= width_delta
    elif original_ratio > rect_ratio:
        # calc deviation
        correct_factor = original_ratio / rect_ratio
        # increase width
        height_delta = int(((height * correct_factor) - height) / 2)
        rectangle_coords['upper'] += height_delta
        rectangle_coords['lower'] -= height_delta

    # check for max shape
    rectangle_coords['left'] = np.max([rectangle_coords['left'], 0])
    rectangle_coords['right'] = np.min([rectangle_coords['right'],
                                        max_shape[1]-1])
    rectangle_coords['lower'] = np.max([rectangle_coords['lower'], 0])
    rectangle_coords['upper'] = np.min([rectangle_coords['upper'],
                                        max_shape[0]-1])
    return rectangle_coords



# Parameters
image_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\MISC2\\'
output_path = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\snapshot_wisconsin\\TEST\\'
algo = 'rectangle_max_mass_proposal'
algo = 'rectangle_max_mass'
algo = 'mixer'
grayscale=True
target_shape = (330, 330)

# get all files in directory
files = sorted(os.listdir(image_path))
files
# create sequence images
image_seq_files = collections.OrderedDict()
for f in files:
    id_name = f.split("_")[0]
    seq_id = f.split("_")[1].split(".")[0]
    if id_name not in image_seq_files:
        image_seq_files[id_name] = collections.OrderedDict()
    image_seq_files[id_name][seq_id] = f

# read and process images
for img_seq in image_seq_files:
    # get all images
    img_dict = dict()
    for ii, img_path in image_seq_files[img_seq].items():
        f_path = image_path + img_path
        # read image from disk
        img = load_img(f_path)
        img_arr = img_to_array(img) / 255
        # store in images_dict
        img_dict[ii] = img_arr
        # calculate average image
        if ii == '0':
            img_avg = np.copy(img_arr)
        else:
            img_avg = img_avg + img_arr
    img_avg = img_avg / len(image_seq_files[img_seq].keys())
    img_dict['avg'] = img_avg

    # calculate blurred versions of all images
    img_blurr_dict = dict()
    for ii, img in img_dict.items():
        if grayscale and (len(img.shape) == 3):
            # convert to grayscale
            if img.shape[2] == 3:
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        blurred = ndimage.gaussian_filter(img, sigma=2)
        img_blurr_dict[ii] = blurred

    # calculate difference images of blurred and average images
    # and determine center of differences
    avg_img = img_blurr_dict['avg']
    points = dict()
    imgs_threshold = dict()

    for ii, img in img_blurr_dict.items():
        if ii == 'avg':
            continue
        # difference between image and average image
        diff = 1-abs(np.squeeze(img) - np.squeeze(avg_img))
        # calculate threshold using isodata algorithm
        thresh = threshold_isodata(diff)
        # create binary image according to thresholded values
        binary = diff < thresh
        binary_img = diff
        binary_img[binary] = 1
        binary_img[binary == False] = 0
        imgs_threshold[ii] = binary_img

        if algo == 'rectangle_mass':
            # calculate center point of difference image
            res = find_max_mass(img=binary_img,
                                window_shape=target_shape, step_size=50)
            res_type = 'rectangle'
            points[ii] = res
        elif algo == 'rectangle_max_mass':
            center = find_max_mass(
                img=binary_img,
                window_shape=(20, 20), step_size=10,
                result="center")
            res_type = 'circle'
            points[ii] = center
        elif algo == 'rectangle_max_mass_proposal':
            res = rectangle_proposal(img=binary_img,
                                     min_rectangle_size=target_shape)
            res_type = 'rectangle'
            points[ii] = res
        elif algo == 'mixer':
            p_motion = motion_share(binary_img)
            # if there is a lot of motion going on use rectangle max
            if p_motion > 0.9:
                center = find_max_mass(
                            img=binary_img,
                            window_shape=(50, 50), step_size=20,
                            result="center")
                res = rectangle_coordinates_from_center(
                    center_x=center[1],
                    center_y=center[0],
                    height=target_shape[0],
                    width=target_shape[1],
                    max_shape=binary_img.shape[0:2])
            # if the pattern is clear use rectangle proposal
            else:
                res = rectangle_proposal(
                    img=binary_img,
                    min_rectangle_size=target_shape)
            res_type = 'rectangle'
            points[ii] = res
        else:
            res_type = 'circle'
            x_sums = np.sum(binary_img, axis=0)
            x_min = np.argmax(x_sums, axis=0)
            y_sums = np.sum(binary_img, axis=1)
            y_min = np.argmax(y_sums, axis=0)
            center = (x_min[0], y_min[0])
            points[ii] = center

    # add shape to original images
    img_dict_shapes = dict()
    for ii, img in img_dict.items():
        if ii == 'avg':
            continue
        if res_type == 'rectangle':
            # calculate and add lines
            rec = points[ii]
            l1 = line(r0=rec['lower'], c0=rec['left'],
                      r1=rec['lower'], c1=rec['right'])
            l2 = line(rec['lower'], rec['left'], rec['upper'], rec['left'])
            l3 = line(rec['upper'], rec['left'], rec['upper'], rec['right'])
            l4 = line(rec['upper'], rec['right'], rec['lower'], rec['right'])
            img_circle = np.copy(img)
            for ll in zip([l1, l2, l3, l4]):
                img_circle[ll[0][0], ll[0][1], 0] = 1
            img_dict_shapes[ii] = img_circle
        elif res_type == 'circle':
            center = points[ii]
            # calculate circle points
            rr, cc = circle(c=center[0], r=center[1], radius=30,
                            shape=img.shape)
            img_circle = np.copy(img)
            img_circle[rr, cc, 0] = 1
            img_dict_shapes[ii] = img_circle

    # crop images from original images
    img_dict_crops = dict()
    for ii, img in img_dict.items():
        if ii == 'avg':
            continue
        if res_type == 'rectangle':
            # crop
            rec = points[ii]
            width = rec['left'] - rec['left']
            height = rec['upper'] - rec['lower']
            img_crop = np.array(img[rec['lower']:rec['upper'],
                                    rec['left']:rec['right'], :])
            # img_crop = np.zeros(shape=(height, width, img.shape[2]))
            img_dict_crops[ii] = img_crop
        elif res_type == 'circle':
            img_dict_crops[ii] = img


    # write image with shape to disk
    for ii, img_path in image_seq_files[img_seq].items():
        if ii == 'avg':
            continue
        # f_path = image_path + img_path
        f_path = output_path + img_path
        f_path_new = f_path.replace(".jpeg", "_shape.jpeg")
        img = img_dict_shapes[ii]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img_to_save = array_to_img(img, )
        # save
        img_to_save.save(f_path_new)

    # write threshold images to disk
    for ii, img_path in image_seq_files[img_seq].items():
        if ii == 'avg':
            continue
        f_path_new = output_path + img_path
        f_path_new = f_path_new.replace(".jpeg", "_threshold.jpeg")
        img = imgs_threshold[ii]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img_to_save = array_to_img(img)
        # save
        img_to_save.save(f_path_new)

    # write cropped image to disk
    for ii, img_path in image_seq_files[img_seq].items():
        if ii == 'avg':
            continue
        f_path_new = output_path + img_path
        f_path_new = f_path_new.replace(".jpeg", "_cropped.jpeg")
        img = img_dict_crops[ii]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img_to_save = array_to_img(img, )
        # save
        img_to_save.save(f_path_new)
