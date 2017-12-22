# Test Motion Detection
import numpy as np
import os
from keras.preprocessing.image import img_to_array,  load_img, array_to_img
from scipy import ndimage
from config.config import cfg_path
from config.config import logging
from skimage.filters import threshold_isodata
# from skimage.draw import (line, circle)
import collections
from sandbox.motion_detection_funcs import\
    (find_max_mass,
     rectangle_coordinates_from_center, motion_share, rectangle_proposal)


def process_images(image_seq_files, image_path, output_path,
                   algo, target_shape, grayscale=True, crop_images=False,
                   combine_images=True):
    # read and process images
    for img_seq in image_seq_files:
        # get all images
        img_dict = dict()
        n_in_seq = 0
        for ii, img_path in image_seq_files[img_seq].items():
            f_path = image_path + img_path
            # read image from disk
            img = load_img(f_path)
            img_arr = img_to_array(img) / 255
            if n_in_seq == 0:
                img_shape = img_arr.shape
            elif img_shape != img_arr.shape:
                img = load_img(f_path, target_size=img_shape)
                img_arr = img_to_array(img) / 255
            # store in images_dict
            img_dict[ii] = img_arr
            # calculate average image
            if ii == '0':
                img_avg = np.copy(img_arr)
            else:
                img_avg = img_avg + img_arr
            n_in_seq += 1
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
            try:
                thresh = threshold_isodata(diff)
                # create binary image according to thresholded values
                binary = diff < thresh
                binary_img = diff
                binary_img[binary] = 1
                binary_img[binary == False] = 0
                imgs_threshold[ii] = binary_img
            except:
                print("Thresholding Image %s failed" % ii)
                binary_img = np.ones(shape=diff.shape)
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
        # img_dict_shapes = dict()
        # for ii, img in img_dict.items():
        #     if ii == 'avg':
        #         continue
        #     if res_type == 'rectangle':
        #         # calculate and add lines
        #         rec = points[ii]
        #         l1 = line(r0=rec['lower'], c0=rec['left'],
        #                   r1=rec['lower'], c1=rec['right'])
        #         l2 = line(rec['lower'], rec['left'], rec['upper'], rec['left'])
        #         l3 = line(rec['upper'], rec['left'], rec['upper'], rec['right'])
        #         l4 = line(rec['upper'], rec['right'], rec['lower'], rec['right'])
        #         img_circle = np.copy(img)
        #         for ll in zip([l1, l2, l3, l4]):
        #             img_circle[ll[0][0], ll[0][1], 0] = 1
        #         img_dict_shapes[ii] = img_circle
        #     elif res_type == 'circle':
        #         center = points[ii]
        #         # calculate circle points
        #         rr, cc = circle(c=center[0], r=center[1], radius=30,
        #                         shape=img.shape)
        #         img_circle = np.copy(img)
        #         img_circle[rr, cc, 0] = 1
        #         img_dict_shapes[ii] = img_circle

        # crop images from original images
        if crop_images:
            img_dict_crops = dict()
            for ii, img in img_dict.items():
                if ii == 'avg':
                    continue
                if res_type == 'rectangle':
                    # crop
                    rec = points[ii]
                    img_crop = np.array(img[rec['lower']:rec['upper'],
                                            rec['left']:rec['right'], :])
                    # img_crop = np.zeros(shape=(height, width, img.shape[2]))
                    img_dict_crops[ii] = img_crop
                elif res_type == 'circle':
                    img_dict_crops[ii] = img

            # write cropped image to disk
            for ii, img_path in image_seq_files[img_seq].items():
                if ii == 'avg':
                    continue
                f_path_new = output_path + img_path
                img = img_dict_crops[ii]
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                img_to_save = array_to_img(img, )
                # save
                img_to_save.save(f_path_new)

        # combine set of motion images to one image
        if combine_images:
            # define output filename of the sequence
            f_path_new = output_path + img_seq + '.jpeg'
            # sequence length
            n_images_in_sequence = len(image_seq_files[img_seq].keys() -
                                       set('avg'))
            dim_image = 0
            counter = 0

            # read all images of a sequence
            for ii, img_path in image_seq_files[img_seq].items():
                if counter == 0:
                    # create numpy array for the sequence
                    img = img_blurr_dict[ii]
                    images_concat = np.zeros(shape=img.shape +
                                             (n_images_in_sequence,))
                counter += 1
                # omit the average image
                if ii == 'avg':
                    continue
                # get current image
                img = img_blurr_dict[ii]
                if len(img.shape) == 3:
                    img = np.squeeze(img)
                # add to images_concat
                images_concat[:, :, dim_image] = img
                dim_image += 1

            # save sequence of image as one image
            img_to_save = array_to_img(images_concat)
            # save
            img_to_save.save(f_path_new)


def run_motion_detection():
    # Parameters
    image_path_overall = cfg_path['images'] + 'all' + os.path.sep
    image_path_output = cfg_path['images'] + 'all_comb' + os.path.sep
    algo = 'mixer'
    grayscale = True
    target_shape = (330, 330)
    seq_length = 3

    # loop over all subfolders
    subfolders = [f.path for f in os.scandir(image_path_overall) if f.is_dir()]

    for dd in subfolders:
        logging.info("Starting with dir %s" % dd)
        image_path = dd + os.path.sep
        output_path = image_path_output + dd.split(os.path.sep)[-1] +\
            os.path.sep
        # create output directory
        if not os.path.exists(output_path):
            logging.debug("Creating %s" % output_path)
            os.mkdir(output_path)

        # get all files in directory
        files = sorted(os.listdir(image_path))

        # create sequence images
        image_seq_files = collections.OrderedDict()
        for f in files:
            id_name = f.split("_")[0]
            seq_id = f.split("_")[1].split(".")[0]
            if id_name not in image_seq_files:
                image_seq_files[id_name] = collections.OrderedDict()
            image_seq_files[id_name][seq_id] = f

        # remove already processed images
        files_done = sorted(os.listdir(output_path))
        image_seq_done = dict()
        for f in files_done:
            id_name = f.split("_")[0]
            seq_id = f.split("_")[1].split(".")[0]
            if id_name not in image_seq_done:
                image_seq_done[id_name] = dict()
            image_seq_done[id_name][seq_id] = f
        for k, v in image_seq_done.items():
            if len(v.keys()) == seq_length:
                image_seq_files.pop(k, None)

        # process image sequences in batches
        all_imgs = list(image_seq_files.keys())
        inter = list(np.arange(0, len(all_imgs), 5))
        inter.append(len(all_imgs)+1)

        for i in range(0, len(inter)-1):
            logging.info("Starting with batch %s/%s" % (i, len(inter)-1))
            current_images = set(all_imgs[inter[i]:inter[i+1]])
            image_seq_files_batch = {k: v for k, v in image_seq_files.items()
                                     if k in current_images}
            # process file sequence
            process_images(image_seq_files=image_seq_files_batch,
                           image_path=image_path,
                           output_path=output_path,
                           algo=algo,
                           target_shape=target_shape,
                           grayscale=grayscale,
                           crop_images=False,
                           combine_images=True)

if __name__ == '__main__':
    run_motion_detection()
