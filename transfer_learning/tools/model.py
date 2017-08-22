"""
Class to implement a Model object
- defines different data sets to train a model on
- defines callbacks for logging options
- defines pre_processing / data augmentation during training
- loads the model infrastructure
- invokes model training functions from Keras
"""
from learning.model_components import create_data_generators, create_callbacks
from learning.model_components import create_optimizer
import time
import os
import importlib
import json
from config.config import cfg_model as cfg, logging
from keras.models import load_model, Model as KerasModel, Sequential
from keras.layers import Dense
from tools.helpers import get_most_rescent_file_with_string
from tools.predictor import Predictor
import dill as pickle


class Model(object):
    """ class to implement a model setup """
    def __init__(self, train_set, test_set, val_set, mod_file,
                 pre_processing, config, cfg_path, num_classes,
                 callbacks=['checkpointer', 'csv_logger', 'tb_logger'],
                 optimizer="sgd"):
        self.test_set = test_set
        self.train_set = train_set
        self.val_set = val_set
        self.mod_file = mod_file
        self.pre_processing = pre_processing
        self.optimizer = optimizer
        self.config = config
        self.cfg_path = cfg_path
        self.callbacks = callbacks
        self.cfg = cfg
        self.num_classes = num_classes
        self.class_mapping = None
        self.datagen_train = None
        self.datagen_test = None
        self.datagen_val = None
        self._opt = None
        self._callbacks_obj = None
        self._model = None
        self.model_dict = None
        self.start_epoch = 0
        self._timestamp = str(cfg['ts'])

        project_id = self.config['projects']['panoptes_id']
        self._model_id = self.config[project_id]['experiment_id']
        self._id = self._model_id + '_' + self._timestamp

    def _loadModel(self):
        """ Load model file """
        # load model file
        logging.info("Loading model file: %s" % self.mod_file)
        self.mod_file = importlib.import_module('learning.models.' +
                                                self.mod_file)

    def _loadOptimizer(self):
        """ load pre defined optimizer """
        logging.info("Loading optimizer %s" % self.optimizer)
        self._opt = create_optimizer(name=self.optimizer)

    def _save(self, postfix=None, create_dir=True):
        """ Save model object and model configs """
        # extract project id for further loading project specifc configs
        project_id = self.config['projects']['panoptes_id']

        # get path to save models
        path_to_save = self.cfg_path['save']

        # define model name to save
        model_id = self.config[project_id]['experiment_id']

        path_to_save = path_to_save.replace("//", "/")

        if postfix is not None:
            out_name = '%s_%s' % (model_id, postfix)
        else:
            out_name = model_id

        # check path
        if not os.path.exists(path_to_save):
            if create_dir:
                os.mkdir(path_to_save)
            else:
                raise FileNotFoundError("Path does not exist")
            
        self.model.save(path_to_save + out_name + '.hdf5')
        logging.info("Saved model %s" % (path_to_save + out_name + '.hdf5'))

        # save model config information
        gen = self.val_generator.image_data_generator

        # Transform some pre-processing statistics such that they can
        # be stored in a json file
        if gen.std is not None:
            std = [float(x) for x in gen.std[0][0]]
        else:
            std = None

        if gen.mean is not None:
            mean = [float(x) for x in gen.mean[0][0]]
        else:
            mean = None

        if gen.principal_components is not None:
            pca = [float(x) for x in gen.principal_components[0][0]]
        else:
            pca = None

        # create cfg dictionary
        model_cfg = {
         'class_mapper': self.class_mapping,
         'pre_processing':
         {
          'rescale': gen.rescale,
          'featurewise_std_normalization': gen.featurewise_std_normalization,
          'featurewise_center': gen.featurewise_center,
          'std': std,
          'mean': mean,
          'principal_components': pca,
          'zca_whitening': gen.zca_whitening,
          'zca_epsilon': gen.zca_epsilon
         }
        }

        # save as json file
        with open(path_to_save + out_name + '_cfg.json', 'w') as fp:
            json.dump(model_cfg, fp)

    def _dataGens(self, target_shape):
        """ generate input data from a generator function that applies
        random / static transformations to the input """

        self.train_generator, self.test_generator,\
            self.val_generator = create_data_generators(self.cfg,
                                                        target_shape,
                                                        self.pre_processing
                                                        )
        # store class mapping (index to class)
        class_mapper = {v: k for k, v in
                        self.train_generator.class_indices.items()}

        self.class_mapping = class_mapper

    def _getCallbacks(self):
        """ create pre-defined callbacks """
        self._callbacks_obj = create_callbacks(self._id,
                                               self.callbacks)

    def _calcClassWeights(self):
        """ calculate class weights according to pre-defined modes """
        if 'class_weight' in self.cfg:
            if self.cfg['class_weights'] == 'none':
                cl_w = None
            elif self.cfg['class_weights'] == 'prop':
                from sklearn.utils.class_weight import compute_class_weight
                ids, labels = self.train_set.getAllIDsLabels()
                classes = self.train_set.labels
                cl_w1 = compute_class_weight(class_weight='balanced',
                                             classes=classes,
                                             y=labels)
                # store in dictionary
                cl_w = {c: w for c, w in zip(classes, cl_w1)}
        else:
            cl_w = None

        return cl_w

    def evaluate(self):
        """ prepare to train/predict with model & save to disk """

        for name in ['test', 'val']:
            predictor = Predictor(
                model=self.model,
                pre_processing=self.test_generator.image_data_generator,
                cfg_model=self.cfg,
                cfg_path=self.cfg_path)

            res = predictor.predict_path(self.cfg_path['images'] + name)

            res.to_csv(self.cfg_path['save'] + self._id + '_preds_' +
                       name + '.csv', index=False)

    def prep_model(self):
        """ prepare model """

        ##################################
        # Model Definition
        ##################################

        # load model
        logging.info("Loading Model")
        self._loadModel()
        self.model_dict = self.mod_file.build_model(self.num_classes)

        # define starting epoch (0 for new models)
        self.start_epoch = 0

        # load already trained model if specified
        if self.cfg['load_model'] not in ('', 'None', None):

            # load latest model
            if self.cfg['load_model'] == 'latest':
                model_file = get_most_rescent_file_with_string(
                    dirpath=self.cfg_path['models'],
                    in_str=self.cfg['experiment_id'],
                    excl_str='best')

            # load specified model from save folder if path is specified
            elif os.sep in self.cfg['load_model']:
                root_save_path = self.cfg_path['save'].split(os.sep)[0:-2]
                root_save_path = os.path.join(os.sep, *root_save_path)
                root_save_path = root_save_path.replace(':', ':' + os.sep)
                model_file = root_save_path + os.sep +\
                    self.cfg['load_model'] + '.hdf5'
            # load model from own models path if only file name is specified
            else:
                model_file = self.cfg_path['models'] +\
                             self.cfg['load_model'] + '.hdf5'

            logging.info("Loading model from disk: %s" % model_file)
            model = load_model(model_file)

            # check if specific layers have to be overwritten with randomly
            # initialized weights, e.g. for transfer-learning / fine-tuning
            if self.cfg['load_model_rand_weights_after_layer'] not in ('',
               'None', None):

                # get randomly initialized model
                model_random = self.model_dict['model']

                # model layer names
                layer_names = [x.name for x in model.layers]

                # layer and all following to randomly initialize
                target_layer = self.cfg['load_model_rand_weights_after_layer']

                # check if target layer is in model
                if target_layer not in layer_names:
                    logging.error("Layer %s not in model.layers" %
                                  target_layer)
                    logging.error("Available Layers %s" %
                                  layer_names)
                    raise IOError("Layer %s not in model.layers" %
                                  target_layer)

                # find layers which have to be kept unchanged
                i_set_random = layer_names.index(target_layer)

                # combine old, trained layers with new random layers
                comb_layers = model.layers[0:i_set_random]
                new_layers = model_random.layers[i_set_random:]
                comb_layers.extend(new_layers)

                # define new model
                new_model = Sequential(comb_layers)

                logging.info("Replacing layers of model with random layers")

                # overwrite model
                model = new_model

                # get optimizer
                self._loadOptimizer()

                # compile model
                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=self._opt,
                              metrics=['accuracy',
                                       'sparse_top_k_categorical_accuracy'])

                # print layers of new model
                for layer, i in zip(model.layers, range(0, len(model.layers))):
                    logging.info("New model - layer %s: %s" %
                                 (i, layer.name))

            # check if output layer has to be replaced
            if self.cfg['load_model_replace_output'] == 1:

                # get model input
                new_input = model.input

                # get old model output before last layer
                old_output = model.layers[-2].output

                # create a new output layer
                new_output = Dense(units=self.num_classes,
                                   kernel_initializer="he_normal",
                                   activation="softmax",
                                   name=model.layers[-1].name)(old_output)

                # combine old model with new output layer
                new_model = KerasModel(inputs=new_input,
                                       outputs=new_output)

                logging.info("Replacing output layer of model")

                # print layers of old model
                for layer, i in zip(model.layers, range(0, len(model.layers))):
                    logging.info("Old model - layer %s: %s" %
                                 (i, layer.name))

                # overwrite model
                model = new_model

                # get optimizer
                self._loadOptimizer()

                # compile model
                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=self._opt,
                              metrics=['accuracy',
                                       'sparse_top_k_categorical_accuracy'])

                # print layers of new model
                for layer, i in zip(model.layers, range(0, len(model.layers))):
                    logging.info("New model - layer %s: %s" %
                                 (i, layer.name))

            # check if layers have to be set to non-trainable
            if self.cfg['load_model_retrain_layer'] not in ('', 'None', None):

                # model layer names
                layer_names = [x.name for x in model.layers]

                # check if layer name is in model
                if not self.cfg['load_model_retrain_layer'] in layer_names:
                    logging.error("Layer %s not in model.layers" %
                                  self.cfg['load_model_retrain_layer'])
                    logging.error("Available Layers %s" %
                                  layer_names)
                    raise IOError("Layer %s not in model.layers" %
                                  self.cfg['load_model_retrain_layer'])

                # look for specified layer and set all previous layers
                # to non-trainable
                n_retrain = layer_names.index(
                                  self.cfg['load_model_retrain_layer']
                                  )
                for layer in model.layers[0:n_retrain]:
                    layer.trainable = False

                logging.info("Setting layers before %s to non-trainable" %
                             self.cfg['load_model_retrain_layer'])

                for layer in model.layers:
                    logging.info("Layer %s is trainable: %s" %
                                 (layer.name, layer.trainable))

            # determine starting epoch if a model was loaded from disk, without
            # changing any layers (continue training)
            if (self.cfg['load_model_retrain_layer'] in
               ('', 'None', None)) and not \
               (self.cfg['load_model_replace_output'] == 1):
                # pick up learning at last epoch
                start_epoch = int(model_file.split('/')[-1].split('_')[-2]) + 1
                logging.info("Pick up training from epoch %s" % start_epoch)

        # create new model and start learning from scratch
        else:
            model = self.model_dict['model']

            # get optimizer
            self._loadOptimizer()

            # compile model
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=self._opt,
                          metrics=['accuracy',
                                   'sparse_top_k_categorical_accuracy'])

        # store model
        self.model = model

        # final model configuration
        logging.info("Final Model Architecture")
        for layer, i in zip(model.layers, range(0, len(model.layers))):
            logging.info("Layer %s: Name: %s Input: %s Output: %s" %
                         (i, layer.name, layer.input_shape,
                          layer.output_shape))

        ##################################
        # Data Generators
        ##################################

        logging.info("Creating Data Generators")
        self._dataGens(target_shape=self.model_dict['input_shape'])

        # save evaluation data generator for predictions
        pickle.dump(self.test_generator.image_data_generator,
                    open(self.cfg_path['models'] + self._id + '_generator.pkl',
                         "wb"))

    def train(self):
        """ train model """

        ##################################
        # Logging
        ##################################

        logging.info("Creating Callbacks")

        # create callbacks
        self._getCallbacks()

        ##################################
        # Class Weights
        ##################################

        logging.info("Calculating Class Weights")
        cl_w = self._calcClassWeights()

        ##################################
        # Training
        ##################################

        logging.info("Starting Training")
        time_s = time.time()
        self.model.fit_generator(
                    self.train_generator,
                    steps_per_epoch=self.train_generator.n //
                    self.cfg['batch_size'],
                    epochs=self.cfg['num_epochs'],
                    workers=10,
                    validation_data=self.val_generator,
                    validation_steps=self.val_generator.n //
                    self.cfg['batch_size'],
                    callbacks=self._callbacks_obj,
                    class_weight=cl_w,
                    use_multiprocessing=bool(self.cfg['multi_processing']),
                    initial_epoch=self.start_epoch)

        logging.info("Finished training after %s minutes" %
                     ((time.time() - time_s) // 60))

        ##################################
        # Load best model
        # (if checkpointing for the best)
        ##################################

        if 'checkpointer_best' in self.callbacks:
            model_file = self.cfg_path['models'] +\
                         self._id + '_' +\
                         "model_best.hdf5"
            self.model.load_weights(model_file)
            logging.info("Loaded weights from %s" % model_file)

        ##################################
        # Evaluation
        ##################################

        logging.info("Starting Evaluation on Validation set")
        # Validation Data
        eval_metrics = self.model.evaluate_generator(
                        self.val_generator,
                        steps=self.val_generator.n // self.cfg['batch_size'],
                        workers=10,
                        use_multiprocessing=bool(self.cfg['multi_processing']))

        # print evaluation
        logging.info("Validation Results")
        for name, value in zip(self.model.metrics_names, eval_metrics):
            logging.info("%s: %s" % (name, value))

        logging.info("Starting Evaluation on Test set")

        # Test Data
        eval_metrics = self.model.evaluate_generator(
                        self.test_generator,
                        steps=self.test_generator.n // self.cfg['batch_size'],
                        workers=10,
                        use_multiprocessing=bool(self.cfg['multi_processing']))

        # print evaluation
        logging.info("Test Results")
        for name, value in zip(self.model.metrics_names, eval_metrics):
            logging.info("%s: %s" % (name, value))

        ##################################
        # Save model to disk
        ##################################

        logging.info("Save model to disk")
        self._save(postfix=self._timestamp)
