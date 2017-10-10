"""
Class to provide a Predictor for applying a model on new images
- takes a folder with image, a model, and model configs as input
"""
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from collections import OrderedDict
import numpy as np
import pandas as pd
import json


class PredictorExternal(object):
    """ Predictor completely independent of the specific
        project infrastructure - to be used for researches who want to apply
        one of the models on their images

    Parameters
    ------------
    path_to_model:
        - path to model file
        - string

    model_cfg_json:
        - path to json with model config
        - Json-string with keys: "class_mapper" & "pre_processing"
        - Optional if keras_datagen provided

    keras_datagen:
        - DataGenerator which will be fit on data using a batch
          of 2'000 randomly selected images, will override
          parameters in model_cfg_json
        - Object of keras.preprocessing.image.ImageDataGenerator
        - Optional if model_cfg_json provided

    class_list:
        - list of classes in order of model output layer
        - list
        - Optional, only needs to be specified if not in model_cfg_json

    refit_on_data:
        - Whether to re-fit the DataGenerator on a batch of
          randomly selected images of the provided data
        - Boolean
        - Default: False (recommended)
    """
    def __init__(self,
                 path_to_model=None,
                 model_cfg_json=None,
                 keras_datagen=None,
                 class_list=None,
                 refit_on_data=False):

        self.path_to_model = path_to_model
        self.keras_datagen = keras_datagen
        self.class_list = class_list
        self.model_cfg_json = model_cfg_json
        self.refit_on_data = refit_on_data
        self.model = None
        self.preds = None
        self.pre_processing = None
        self.color_mode = "rgb"

        # Checks
        if path_to_model is None:
            raise IOError("Path to model has to be specified")

        if model_cfg_json is None:
            if keras_datagen is None:
                raise IOError("Specify keras ImageDataGenerator or model cfg")

            if class_list is None:
                raise IOError("Specify class list to map predictions\
                               to classes")

        if model_cfg_json is not None:
            if (keras_datagen is not None) or (class_list is not None):
                raise IOError("Specify only one of model_cfg_json or\
                               (keras_datagen and class_list)\
                               class_list should be in model cfg json")

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
            assert 'class_mapper' in model_cfg.keys(),\
                "class_mapper not found in model_cfg_json,\
                 following keys found %s" % model_cfg.keys()

            assert 'pre_processing' in model_cfg.keys(),\
                "pre_processing not found in model_cfg_json,\
                 following keys found %s" % model_cfg.keys()

            # extract class mapping and order
            class_list = list()
            for i in range(0, len(model_cfg['class_mapper'].keys())):
                class_list.append(model_cfg['class_mapper'][str(i)])
                self.class_list = class_list

            # add pre_processing
            self.pre_processing = model_cfg['pre_processing']

    def predict_path(self, path, output_path,
                     output_file_name='predictions.csv'
                     batch_size=256):
        """ Predict class for images

            Parameters
            ------------
            path:
                - path to directory that contains 1:N sub-directories
                  with images
                - string

            output_path:
                - path to directory to which prediction csv will be written
                - string

            output_file_name:
                - file name of the output csv written to output_path
                - string

            batch_size:
                - number of images to process in one batch, if too large it
                  might not fit into memory
                - integer
        """

        # check input
        if any([x is None for x in [path, output_path]]):
            raise IOError("Path and output_path have to be specified")

        # check output_path
        if not output_path[-1] in ('/', '\\'):
            output_path = output_path + os.path.sep

        # check batch_size
        assert type(eval(batch_size)) == int,\
            "batch_size has to be an integer, is %s" % type(eval(batch_size))

        # fit data generator on input data
        if self.pre_processing is None:

            print("Initializing generator")
            generator = self.keras_datagen.flow_from_directory(
                    path,
                    target_size=self.model.input_shape[1:3],
                    color_mode=self.color_mode,
                    batch_size=batch_size,
                    class_mode='sparse',
                    seed=123,
                    shuffle=False)

            # Fit data generator if required
            if any([self.keras_datagen.featurewise_std_normalization,
                    self.keras_datagen.samplewise_std_normalization,
                    self.keras_datagen.zca_whitening]):

                self._refit_datagen(path, self.keras_datagen)

        # use pre-defined pre_processing options and add to generator
        else:
            print("Initializing generator")
            gen = ImageDataGenerator(rescale=1./255)

            if self.refit_on_data:
                self._refit_datagen(path, gen)
            else:
                # set pre-processing attributes
                for k, v in self.pre_processing.items():
                    if type(v) is list:
                        v = np.array(v)
                    setattr(gen, k, v)

            generator = gen.flow_from_directory(
                    path,
                    target_size=self.model.input_shape[1:3],
                    color_mode=self.color_mode,
                    batch_size=batch_size,
                    class_mode='sparse',
                    seed=123,
                    shuffle=False)

        # predict whole set
        print("Starting to predict images in path")

        # calculate number of iterations to make
        steps_remainder = generator.n % batch_size
        if steps_remainder > 0:
            extra_step = 1
        else:
            extra_step = 0

        preds = self.model.predict_generator(
            generator,
            steps=(generator.n // batch_size) + extra_step,
            workers=1,
            use_multiprocessing=False,
            verbose=1)

        print("Finished predicting %s of %s images" %
              (preds.shape[0], generator.n))
        # check size and log critical
        if preds.shape[0] != generator.n:
            print("Number of Preds %s don't match" +
                  "number of images %s" % (preds.shape[0], generator.n))

        # save predictions
        self.preds = preds

        # Create a data frame with all predictions
        print("Creating Result DF")
        res = self._create_result_df(generator.filenames,
                                     generator.directory)

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
                image_path = image_directory + class_dir +\
                             os.path.sep + fname

            res[i] = OrderedDict([('file_name', fname),
                                  ('predicted_class', y_pred),
                                  ('predicted_probability', p),
                                  ('predictions_all', preds_all),
                                  ('image_path', image_path)])

        res_df = pd.DataFrame.from_dict(res, orient="index")

        return res_df

    def _refit_datagen(self, path, datagen):
        """ Fit Datagenerator on Raw Images """
        print("Fitting data generator")
        # create a generator to randomly select images to calculate
        # image statistics for data pre-processing
        datagen_raw = ImageDataGenerator(rescale=1./255)
        raw_generator = datagen_raw.flow_from_directory(
                path,
                target_size=self.model.input_shape[1:3],
                color_mode=self.color_mode,
                batch_size=2000,
                class_mode='sparse',
                seed=123,
                shuffle=True)
        # fit the generator with a batch of sampled data
        X_raw, Y_raw = raw_generator.next()
        datagen.fit(X_raw)


if __name__ == '__main__':
    model_file = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/models/3663/mnist_testing_201708141308_model_best.hdf5"
    pre_processing = ImageDataGenerator(rescale=1./255)
    pred_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown"
    output_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/save/3663/"
    class_list = [str(i) for i in range(0,10)]

    predictor = PredictorExternal(
        path_to_model=model_file,
        keras_datagen=pre_processing,
        class_list=class_list)

    predictor.predict_path(path=pred_path, output_path=output_path)

    from config.config import cfg_path
    model_cfg_json = cfg_path['save'] + 'cc_species_v2_201708210308_cfg.json'
    model_cfg_json
    cfg_file = open(model_cfg_json, 'r')
    model_cfg = json.load(cfg_file)
    model_cfg.keys()
    model_cfg['pre_processing']
