"""
Class to implement a Predictor
- takes a subject set or a single image and predicts classes
"""
import os
from config.config import cfg_model, cfg_path, logging
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd


class Predictor(object):
    """ class to implement a model setup """
    def __init__(self,
                 cfg_model, cfg_path, mod_file=None,
                 pre_processing=None, model=None):
        self.mod_file = mod_file
        self.pre_processing = pre_processing
        self.cfg_model = cfg_model
        self.cfg_path = cfg_path
        self.model = model
        self.datagen = None
        self.color_mode = None

        if all([x is None for x in [self.mod_file, self.model]]):
            IOError("either mod_file or model has to be not None")

        # load model from disk
        if self.model is None:
            # build path to model file
            model_file = (self.cfg_path['models'] +
                          self.mod_file + '.hdf5')

            # check if model file exists
            if not os.path.isfile(model_file):
                FileNotFoundError("Model File %s not found" % model_file)

            # load model from disk
            logging.info("Loading model from disk: %s" % model_file)
            self.model = load_model(model_file)

        # initialize Pre-Processing
        if self.pre_processing is None:
            self.datagen = ImageDataGenerator(
                rescale=1./255)
        else:
            self.datagen = self.pre_processing

        # handle color mode
        if self.model.input_shape[3] == 1:
            self.color_mode = 'grayscale'
        else:
            self.color_mode = 'rgb'

    def _create_result_df(self, preds, file_names,
                          y_true, class_indices,
                          image_links=""):

        # get max predictions & class ids
        id_max = np.argmax(preds, axis=1)
        max_pred = np.amax(preds, axis=1)

        # map class names and indices
        class_mapper = {v: k for k, v in class_indices.items()}

        # create result dictionary
        res = pd.DataFrame(columns=('subject_id', 'image_id', 'y_true',
                                    'y_pred', 'p', 'link', 'model'))

        for i in range(0, len(file_names)):
            if '\\' in file_names[i]:
                subject_id = file_names[i].split('\\')[1].split('_')[0]
            else:
                subject_id = file_names[i].split('/')[1].split('_')[0]
            image_id = file_names[i].split('_')[1].split('.')[0]
            p = max_pred[i]
            y_pred = class_mapper[id_max[i]]
            if image_links == '':
                link = ''
            else:
                link = image_links[i]
            res.loc[i] = [subject_id, image_id, class_mapper[y_true[i]],
                          y_pred, p,
                          link, self.mod_file]

        return res

    def predict_path(self, path):
        """ Predict path with class folders """

        logging.info("Initializing generator")
        generator = self.datagen.flow_from_directory(
                path,
                target_size=self.model.input_shape[1:3],
                color_mode=self.color_mode,
                batch_size=256,
                class_mode='sparse',
                seed=self.cfg_model['random_seed'],
                shuffle=False)

        # predict whole set
        logging.info("Predicting images in path")
        preds = self.model.predict_generator(
            generator,
            steps=(generator.n // 256) + 1,
            workers=1,
            use_multiprocessing=bool(self.cfg_model['multi_processing']))

        # consolidate output
        logging.info("Creating Result DF")
        res = self._create_result_df(preds, generator.filenames,
                                     generator.classes,
                                     generator.class_indices,
                                     image_links="")

        return res


if __name__ == '__main__':
    pass


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
