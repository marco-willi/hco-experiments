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
