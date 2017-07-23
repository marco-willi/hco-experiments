"""
Class to implement a Model object
- defines different data sets to train a model on
- defines callbacks for logging options
- defines pre_processing / data augmentation during training
- loads the model infrastructure
- invokes model training functions from Keras
"""
from learning.helpers import create_data_generators, create_callbacks
from learning.helpers import create_optimizer
import time
import os
import importlib
from config.config import cfg_model as cfg, logging
from keras.models import load_model
from datetime import datetime
from tools.helpers import get_most_rescent_file_with_string


class Model(object):
    """ class to implement a model setup """
    def __init__(self, train_set, test_set, val_set, mod_file,
                 pre_processing, config, cfg_path, num_classes,
                 callbacks=['checkpointer', 'csv_logger', 'tb_logger'],
                 optimizer="standard"):
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
        self.datagen_train = None
        self.datagen_test = None
        self.datagen_val = None
        self._opt = None
        self._callbacks_obj = None
        self._model = None
        self._timestamp = datetime.now().strftime('%Y%m%d%H%m')

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
        """ Save model object """
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
        if not os.path.exists(path_to_save) & create_dir:
            os.mkdir(path_to_save)
        else:
            NameError("Path not Found")

        self.model.save(path_to_save + out_name + '.h5')
        logging.info("Saved model %s" % (path_to_save + out_name + '.h5'))

    def _dataGens(self, target_shape):
        """ generate input data from a generator function that applies
        random / static transformations to the input """

        self.train_generator, self.test_generator,\
            self.val_generator = create_data_generators(self.cfg,
                                                        target_shape,
                                                        self.pre_processing
                                                        )

    def _getCallbacks(self):
        """ create pre-defined callbacks """
        project_id = self.config['projects']['panoptes_id']
        model_id = self.config[project_id]['experiment_id']
        self._callbacks_obj = create_callbacks(model_id + '_' +
                                               self._timestamp,
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

    def train(self):
        """ train model """

        ##################################
        # Model Definition
        ##################################

        # load model
        logging.info("Loading Model")
        self._loadModel()
        model_dict = self.mod_file.build_model(self.num_classes)

        # define starting epoch (0 for new models)
        start_epoch = 0

        # load model if specified
        if self.cfg['load_model'] not in ('', 'None', None):

            # load latest model
            if self.cfg['load_model'] == 'latest':
                model_file = get_most_rescent_file_with_string(
                    dirpath=self.cfg_path['models'],
                    in_str=self.cfg['experiment_id'],
                    excl_str='best')

            # load specified model
            else:
                model_file = self.cfg_path['models'] +\
                             self.cfg['load_model'] + '.hdf5'

            logging.info("Loading model from disk: %s" % model_file)
            model = load_model(model_file)

            # pick up learning at last epoch
            start_epoch = int(model_file.split('/')[-1].split('_')[-2]) + 1

            logging.info("Pick up training from epoch %s" % start_epoch)

        # create new model and start learning from scratch
        else:
            model = model_dict['model']

            # get optimizer
            self._loadOptimizer()

            # compile model
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=self._opt,
                          metrics=['accuracy',
                                   'sparse_top_k_categorical_accuracy'])

        # store model
        self.model = model

        ##################################
        # Data Generators
        ##################################

        logging.info("Creating Data Generators")
        self._dataGens(target_shape=model_dict['input_shape'])

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
                    validation_data=self.test_generator,
                    validation_steps=self.test_generator.n //
                    self.cfg['batch_size'],
                    callbacks=self._callbacks_obj,
                    class_weight=cl_w,
                    use_multiprocessing=bool(self.cfg['multi_processing']),
                    initial_epoch=start_epoch)

        print("Finished training after %s minutes" %
              ((time.time() - time_s) // 60))

        ##################################
        # Evaluation
        ##################################

        # Test Data
        eval_metrics = model.evaluate_generator(
                        self.test_generator,
                        steps=self.test_generator.n // self.cfg['batch_size'],
                        workers=10,
                        use_multiprocessing=bool(self.cfg['multi_processing']))

        # print evaluation
        print("Test Results")
        for name, value in zip(self.model.metrics_names, eval_metrics):
            print("%s: %s" % (name, value))

        # Validation Data
        eval_metrics = self.model.evaluate_generator(
                        self.val_generator,
                        steps=self.val_generator.n // self.cfg['batch_size'],
                        workers=10,
                        use_multiprocessing=bool(self.cfg['multi_processing']))

        # print evaluation
        print("Validation Results")
        for name, value in zip(self.model.metrics_names, eval_metrics):
            print("%s: %s" % (name, value))

        ##################################
        # Save model to disk
        ##################################

        self._save(postfix=self._timestamp)
