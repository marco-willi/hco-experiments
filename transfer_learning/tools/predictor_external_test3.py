"""
Class to implement a Predictor for applying a model on new images
- takes a folder with images and a model as input
- additionally requires pre-processing specification and class list
"""
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, list_pictures, load_img, img_to_array
# from tools.double_iterator import DoubleIterator
from collections import OrderedDict
import numpy as np
import pandas as pd
import json
import keras.backend as K


def calc_crops(input_shape, crop_shape):
    """ Calc different crop windows """
    assert len(input_shape) == 2, "input_shape has to be a tuple of length 2"
    assert len(crop_shape) == 2, "crop_shape has to be a tuple of length 2"
    assert input_shape[0] > crop_shape[0],\
        "input_shape smaller than crop_shape"
    assert input_shape[1] > crop_shape[1],\
        "input_shape smaller than crop_shape"

    res = dict()

    # center crop
    start_x = (input_shape[0] // 2) - (crop_shape[0] // 2)
    start_y = (input_shape[1] // 2) - (crop_shape[1] // 2)
    end_x = start_x + crop_shape[0]
    end_y = start_y + crop_shape[1]
    res['center'] = {"start_x": start_x, "end_x": end_x,
                     "start_y": start_y, "end_y": end_y}

    # top left crop
    start_x = 0
    start_y = 0
    end_x = start_x + crop_shape[0]
    end_y = start_y + crop_shape[1]
    res['top_left'] = {"start_x": start_x, "end_x": end_x,
                       "start_y": start_y, "end_y": end_y}

    # top right crop
    start_x = 0
    end_y = input_shape[1]
    end_x = start_x + crop_shape[0]
    start_y = end_y - crop_shape[1]
    res['top_right'] = {"start_x": start_x, "end_x": end_x,
                        "start_y": start_y, "end_y": end_y}

    # bottom left crop
    end_x = input_shape[0]
    start_y = 0
    end_y = start_y + crop_shape[1]
    start_x = end_x - crop_shape[0]
    res['bottom_left'] = {"start_x": start_x, "end_x": end_x,
                          "start_y": start_y, "end_y": end_y}

    # bottom right crop
    end_x = input_shape[0]
    end_y = input_shape[1]
    start_y = end_y - crop_shape[1]
    start_x = end_x - crop_shape[0]
    res['bottom_right'] = {"start_x": start_x, "end_x": end_x,
                           "start_y": start_y, "end_y": end_y}

    return res


def crop_array(X, x1, x2, y1, y2):
    """ crop array """
    if len(X.shape) == 4:
        return X[:, x1:x2, y1:y2, :]
    else:
        return X[x1:x2, y1:y2, :]


def center_crop(X, crop_size):
    assert len(crop_size) == 2, "crop_size has to be a tuple of length 2"
    orig_shape = X.shape
    if len(X.shape) == 4:
        orig_shape = X.shape[1:3]
    else:
        orig_shape = X.shape[0:2]

    if crop_size[0] >= orig_shape[0]:
        start_x = 0
    else:
        start_x = (orig_shape[0] // 2) - (crop_size[0] // 2)

    if crop_size[1] >= orig_shape[1]:
        start_y = 0
    else:
        start_y = (orig_shape[1] // 2) - (crop_size[1] // 2)

    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]

    if len(X.shape) == 4:
        return X[:, start_x:end_x, start_y:end_y, :]
    else:
        return X[start_x:end_x, start_y:end_y, :]

def center_crop(X, crop_size):
    assert len(crop_size) == 2, "crop_size has to be a tuple of length 2"
    orig_shape = X.shape
    if len(X.shape) == 4:
        orig_shape = X.shape[1:3]
    else:
        orig_shape = X.shape[0:2]

    if crop_size[0] >= orig_shape[0]:
        start_x = 0
    else:
        start_x = (orig_shape[0] // 2) - (crop_size[0] // 2)

    if crop_size[1] >= orig_shape[1]:
        start_y = 0
    else:
        start_y = (orig_shape[1] // 2) - (crop_size[1] // 2)

    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]

    if len(X.shape) == 4:
        return X[:, start_x:end_x, start_y:end_y, :]
    else:
        return X[start_x:end_x, start_y:end_y, :]

class PredictorExternal(object):
    """ class to implement a predictor, completely independent of the specific
        project infrastructure - to be used for researches who want to apply
        one of the models
    """
    def __init__(self,
                 path_to_model=None,
                 model_cfg_json=None,
                 keras_datagen=None,
                 class_list=None,
                 pred_mode="standard"):

        self.path_to_model = path_to_model
        self.keras_datagen = keras_datagen
        self.class_list = class_list
        self.model_cfg_json = model_cfg_json
        self.model = None
        self.preds = None
        self.pre_processing = None
        self.color_mode = "rgb"
        self.pred_mode = pred_mode

        if path_to_model is None:
            raise IOError("Path to model has to be specified")

        if model_cfg_json is None:
            if keras_datagen is None:
                raise IOError("Specify keras ImageDataGenerator")

            if class_list is None:
                raise IOError("Specify class list to map predictions\
                               to classes")

        if model_cfg_json is not None:
            if (keras_datagen is not None) or (class_list is not None):
                raise IOError("Specify only one of model_cfg_json or\
                               (keras_datagen and class_list)")

        if not os.path.isfile(self.path_to_model):
            raise FileNotFoundError("Model File %s not found" %
                                    self.path_to_model)

        # Load model from disk
        print("Loading model from disk: %s" % self.path_to_model)
        self.model = load_model(self.path_to_model)

        # handle color mode
        if self.model.input_shape[3] == 1:
            self.color_mode = 'grayscale'
        else:
            self.color_mode = 'rgb'

        # Load cfg from json file
        if model_cfg_json is not None:
            cfg_file = open(model_cfg_json, 'r')
            model_cfg = json.load(cfg_file)
            # check model_cfg
            assert 'class_mapper' in model_cfg.keys()
            assert 'pre_processing' in model_cfg.keys()

            # extract class mapping and order
            class_list = list()
            for i in range(0, len(model_cfg['class_mapper'].keys())):
                class_list.append(model_cfg['class_mapper'][str(i)])
                self.class_list = class_list

            # add pre_processing
            self.pre_processing = model_cfg['pre_processing']

    def predict_path(self, path, output_path,
                     output_file_name='predictions.csv'):
        """ Predict class for images in a directory with subdirectories that
            contain the images """

        # check input
        if any([x is None for x in [path, output_path]]):
            raise IOError("Path and output_path have to be specified")

        # check output_path
        if not output_path[-1] in ('/', '\\'):
            output_path = output_path + os.path.sep

        # prediction batch sizes
        batch_size = 3

        # fit data generator on input data
        if self.pre_processing is None:

            print("Initializing generator")
            generator = self.keras_datagen.flow_from_directory(
                    path,
                    target_size=self.model.input_shape[1:3],
                    color_mode=self.color_mode,
                    batch_size=batch_size,
                    seed=123,
                    shuffle=False)

            # Fit data generator if required
            if any([self.keras_datagen.featurewise_std_normalization,
                    self.keras_datagen.samplewise_std_normalization,
                    self.keras_datagen.zca_whitening]):

                print("Fitting data generator")
                # create a generator to randomly select images to calculate
                # image statistics for data pre-processing
                generator_fit = self.keras_datagen.flow_from_directory(
                        path,
                        target_size=self.model.input_shape[1:3],
                        color_mode=self.color_mode,
                        batch_size=2000,
                        seed=123,
                        shuffle=True)

                # fit the generator with a batch of sampled data
                generator.fit(generator_fit.next())

        # use pre-defined pre_processing options and add to generator
        else:
            print("Initializing generator")
            gen = ImageDataGenerator(rescale=1./255)

            for k, v in self.pre_processing.items():
                if type(v) is list:
                    v = np.array(v)
                setattr(gen, k, v)

        # predict whole set
        print("Starting to predict images in path")

        # find all images
        image_paths = list_pictures(path)

        # calculate number of iterations to make
        steps_remainder = len(image_paths) % batch_size
        if steps_remainder > 0:
            extra_step = 1
        else:
            extra_step = 0

        n_batches = (len(image_paths) // batch_size) + extra_step

        # prediction mode
        if self.pred_mode == "standard":
            target_size = self.model.input_shape[1:]
        elif self.pred_mode == "center_crop":
            target_size = list(self.model.input_shape[1:])
            target_size[0] = target_size[0] * 2
            target_size[1] = target_size[1] * 2
            target_size = tuple(target_size)
        elif self.pred_mode == "5_crop":
            target_size = list(self.model.input_shape[1:])
            target_size[0] = target_size[0] * 2
            target_size[1] = target_size[1] * 2
            target_size = tuple(target_size)

        # get batches of data from disk

        preds = np.zeros(shape=(len(image_paths), len(self.class_list)))

        for step in range(0, n_batches):
            idx_start = step * batch_size
            idx_end = np.min([idx_start + batch_size, preds.shape[0]])

            batch_x = np.zeros((idx_end-idx_start,) + target_size,
                                dtype=K.floatx())

            for i, idx in enumerate(range(idx_start, idx_end)):
                grayscale = self.color_mode == 'grayscale'
                img = load_img(image_paths[idx],
                               grayscale=grayscale,
                               target_size=target_size)
                img_x = img_to_array(img, data_format=K.image_data_format())
                img_x = gen.standardize(img_x)
                batch_x[i] = img_x

            # predict on batch
            if self.pred_mode == "standard":
                p_batch = self.model.predict_on_batch(batch_x)
            elif self.pred_mode == "center_crop":
                crops = calc_crops(input_shape=target_size[0:2],
                                   crop_shape=self.model.input_shape[1:3])
                crop = crops["center"]
                batch_x_cropped = crop_array(batch_x, x1=crop["start_x"],
                                             x2=crop["end_x"],
                                             y1=crop["start_y"],
                                             y2=crop["end_y"])

                p_batch = self.model.predict_on_batch(batch_x_cropped)

            elif self.pred_mode == "5_crop":
                crops = calc_crops(input_shape=target_size[0:2],
                                   crop_shape=self.model.input_shape[1:3])

                for n, crop in crops.items():
                    batch_x_cropped = crop_array(batch_x, x1=crop["start_x"],
                                                 x2=crop["end_x"],
                                                 y1=crop["start_y"],
                                                 y2=crop["end_y"])

                    p_batch = self.model.predict_on_batch(batch_x_cropped)

            preds[idx_start: idx_end, :] = p_batch

        print("Finished predicting %s of %s images" %
              (preds.shape[0], len(image_paths)))
        # check size and log critical
        if preds.shape[0] != len(image_paths):
            print("Number of Preds %s don't match" +
                  "number of images %s" % (preds.shape[0], len(image_paths)))

        # save predictions
        self.preds = preds

        # Create a data frame with all predictions
        print("Creating Result DF")
        res = self._create_result_df(image_paths)

        # write DF to disk
        res.to_csv(output_path + output_file_name, index=False)

    def _create_result_df(self, filenames,
                          image_directory=""):
        """ Create Data Frame with Predictions """

        # get max predictions & class ids
        id_max = np.argmax(self.preds, axis=1)
        max_pred = np.amax(self.preds, axis=1)

        # map class names and indices
        n_classes = len(self.class_list)

        # create result data frame via dictionary
        res = OrderedDict()

        # loop over all files / predictions
        for i in range(0, len(filenames)):
            fname = filenames[i].split(os.path.sep)[1]
            class_dir = filenames[i].split(os.path.sep)[0]

            p = max_pred[i]
            y_pred = self.class_list[id_max[i]]

            # store predictions for all classes
            p_all = self.preds[i, :]
            preds_all = {self.class_list[j]: p_all[j] for j in
                         range(0, n_classes)}

            if image_directory == '':
                image_path = ''
            else:
                image_path = image_directory + os.path.sep + class_dir +\
                             os.path.sep + fname

            res[i] = OrderedDict([('file_name', fname),
                                  ('predicted_class', y_pred),
                                  ('predicted_probability', p),
                                  ('predictions_all', preds_all),
                                  ('image_path', image_path)])

        res_df = pd.DataFrame.from_dict(res, orient="index")

        return res_df


if __name__ == '__main__':
    pass
    from config.config import cfg_path
    # model_file = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/models/3663/mnist_testing_201708141308_model_best.hdf5"
    # pre_processing = ImageDataGenerator(rescale=1./255)
    # pred_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown"
    # output_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/save/3663/"
    # model_cfg_json = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\save\\3663\\mnist_testing_201708141308_cfg.json"
    # class_list = [str(i) for i in range(0,10)]

    model_file = cfg_path['save'] + 'cc_species_v2_201708210308.hdf5'
    model_cfg_json = cfg_path['save'] + 'cc_species_v2_201708210308_cfg.json'
    pred_path = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\camera_catalogue\\exp_south_africa_4\\unknown\\"
    output_path = cfg_path['save']
    output_file_name='predictions_TEST.csv'

    predictor = PredictorExternal(
        path_to_model=model_file,
        model_cfg_json=model_cfg_json, pred_mode="5_crop")

    predictor.predict_path(path=pred_path, output_path=output_path,
                           output_file_name=output_file_name)


    # model_cfg_json = cfg_path['save'] + 'cc_species_v2_201708210308_cfg.json'
    # model_cfg_json
    # cfg_file = open(model_cfg_json, 'r')
    # model_cfg = json.load(cfg_file)
    # model_cfg.keys()
    # model_cfg['pre_processing']

    # preds.shape
    # preds.shape
    # generator.directory
    # generator.class_indices
    # generator.filenames
    # model.summary()



    # model_file = cfg_path['models'] +\
    #              'cat_vs_dog_testing_201707111307_model_best' + '.hdf5'
    #
    # model_file
    # model = load_model(model_file)
    # model.get_config()
    # model





    # datagen = ImageDataGenerator(
    #     rescale=1./255)
    #
    # generator = datagen.flow_from_directory(
    #         cfg_path['images'] + 'val',
    #         target_size=model.input_shape[1:3],
    #         color_mode='rgb',
    #         batch_size=256,
    #         class_mode='sparse',
    #         seed=123,
    #         shuffle=False)
    # preds = model.predict_generator(
    #     generator,
    #     steps=(generator.n // 256)+1,
    #     workers=2)
    # preds.shape
    # preds[0:5,:]
    # generator.


    # predictor = Predictor(mod_file='cat_vs_dog_testing_201707111307_model_best',
    #                       cfg_model=cfg_model,
    #                       cfg_path=cfg_path)
    #
    # res = predictor.predict_path(cfg_path['images'] + 'val')
    #
    # res.head


    # model_file
    # model = load_model(model_file)
    # model.get_config()
    # predictor = Predictor(mod_file='mnist_testing_201707231307_model_best',
    #                       cfg_model=cfg_model,
    #                       cfg_path=cfg_path)
    #
    # res = predictor.predict_dir(cfg_path['images'] + 'val')
    #
    # res.head

    # preds, nams, cl, cl_i = predictor.predict_dir(cfg_path['images'] + 'val')
    #
    # import numpy as np
    # import pandas as pd
    # id_max = np.argmax(preds, axis=1)
    # max_pred = np.amax(preds, axis=1)
    # y_true = cl
    # class_mapper = {v: k for k, v in cl_i.items()}
    #
    # # create result dictionary
    # res = pd.DataFrame(columns=('subject_id', 'image_id', 'y_true',
    #                             'y_pred', 'p'))
    # i=0
    # for i in range(0, len(nams)):
    #     subject_id = nams[i].split('\\')[1].split('_')[0]
    #     image_id = nams[i].split('_')[1].split('.')[0]
    #     p = max_pred[i]
    #     y_true = cl[i]
    #     y_pred = class_mapper[id_max[i]]
    #     res.loc[i] = [subject_id, image_id, y_true, y_pred, p]
    #
    # res.head
    #
    #
    #
    #
    #
    #
    # tt = np.amax(preds, axis=1)
    # tt.shape
    # tt[0:10]
    #
    # preds.shape
    # preds[0:5,:]
    # nams[0:5]
    # nams[0]
    # class_mapper
    # cl_i
