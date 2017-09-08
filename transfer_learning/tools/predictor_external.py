"""
Class to implement a Predictor for applying a model on new images
- takes a folder with images and a model as input
- additionally requires pre-processing specification and class list
"""
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import sys


class PredictorExternal(object):
    """ class to implement a predictor, completely independent of the specific
        project infrastructure - to be used for researches who want to apply
        one of the models
    """
    def __init__(self,
                 path_to_model=None,
                 keras_datagen=None,
                 class_list=None):

        self.path_to_model = path_to_model
        self.keras_datagen = keras_datagen
        self.class_list = class_list
        self.model = None
        self.preds = None
        self.color_mode = "rgb"

        if path_to_model is None:
            raise IOError("Path to model has to be specified")

        if keras_datagen is None:
            raise IOError("Specify keras ImageDataGenerator")

        if class_list is None:
            raise IOError("Specify class list to map predictions to classes")

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

    def predict_path(self, path, output_path):
        """ Predict class for images in a directory with subdirectories that
            contain the images """

        # check input
        if any([x is None for x in [path, output_path]]):
            raise IOError("Path and output_path have to be specified")

        # check output_path
        if output_path[-1] != os.path.sep:
            output_path = output_path + os.path.sep

        # prediction batch sizes
        batch_size = 256

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

            print("Fitting data generator")
            # create a generator to randomly select images to calculate
            # image statistics for data pre-processing
            generator_fit = self.keras_datagen.flow_from_directory(
                    path,
                    target_size=self.model.input_shape[1:3],
                    color_mode=self.color_mode,
                    batch_size=2000,
                    class_mode='sparse',
                    seed=123,
                    shuffle=True)

            # fit the generator with a batch of sampled data
            generator.fit(generator_fit.next())

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
            use_multiprocessing=False)

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
        res.to_csv(output_path + 'predictions.csv', index=False)

    def _create_result_df(self, filenames,
                          image_directory=""):
        """ Create Data Frame with Predictions """

        # get max predictions & class ids
        id_max = np.argmax(self.preds, axis=1)
        max_pred = np.amax(self.preds, axis=1)

        # map class names and indices
        n_classes = len(self.class_list)

        # determine platform for output file path format
        platform = sys.platform

        # create result data frame
        res = pd.DataFrame(columns=('file_name', 'predicted_class',
                                    'predicted_probability', 'predictions_all',
                                    'image_path'))

        # loop over all files / predictions
        for i in range(0, len(filenames)):
            if '\\' in filenames[i]:
                fname = filenames[i].split('\\')[1]
                class_dir = filenames[i].split('\\')[0]
            else:
                fname = filenames[i].split('/')[1]
                class_dir = filenames[i].split('/')[0]

            p = max_pred[i]
            y_pred = self.class_list[id_max[i]]

            # store predictions for all classes
            p_all = self.preds[i, :]
            preds_all = {self.class_list[j]: p_all[j] for j in
                         range(0, n_classes)}

            if image_directory == '':
                image_path = ''
            else:
                image_path = image_directory + "/" + class_dir + "/" + fname

            # replace forward slash on windows platforms
            if platform == 'win32':
                image_path.replace('/', '\\')

            res.loc[i] = [fname, y_pred, p, preds_all,
                          image_path]

        return res


if __name__ == '__main__':
    pass

    model_file = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/models/3663/mnist_testing_201708021008_model_04_0.40.hdf5"
    pre_processing = ImageDataGenerator(rescale=1./255)
    pred_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown"
    output_path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/images/3663/unknown/"
    class_list = [str(i) for i in range(0,10)]

    predictor = PredictorExternal(
        path_to_model=model_file,
        keras_datagen=pre_processing,
        class_list=class_list)

    predictor.predict_path(path=pred_path, output_path=output_path)


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
