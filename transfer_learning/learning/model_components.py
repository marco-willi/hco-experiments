"""
Implementation of different Options / Parameters for model definitions
- create_class_mappings
- create_optimizer
- create_data_generators
- create_callbacks
- model_save
"""
# from tools.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from config.config import config, cfg_path, cfg_model, logging
import os
from keras.optimizers import SGD, Adagrad, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, RemoteMonitor, Callback
from subprocess import run, PIPE
from keras import backend as K
import numpy as np
from tools.double_iterator import DoubleIterator


# create class mappings
def create_class_mappings(mapping="1_on_1", excl_classes=None,
                          only_these_classes=None):
    """ Creates class mappings according to pre-defined options """
    # all classes
    cfg = cfg_model
    all_classes = cfg['classes']

    # blank classes
    blank_classes = ['blank', 'NOTHINGHERE']

    if excl_classes is not None:
        all_classes = [x for x in all_classes if x not in excl_classes]

    if only_these_classes is not None:
        all_classes = [x for x in all_classes if x in only_these_classes]

    # default 1 to 1 mapping
    if mapping == "1_on_1":
        map_dict = {c: c for c in all_classes}

    # blank vs non-blank
    elif mapping == "blank_vs_nonblank":
        map_dict = dict()
        for c in all_classes:
            if c in blank_classes:
                map_dict[c] = 'blank'
            else:
                map_dict[c] = 'non_blank'

    # non blanks
    elif mapping == "nonblank":
        map_dict = {c: c for c in all_classes}
        for bl in blank_classes:
            map_dict.pop(bl, None)

    elif mapping == "ss_nonblank":
        excl_classes = ['bat', 'cattle', 'steenbok']
        all_classes = [x for x in all_classes if x not in excl_classes]
        map_dict = {c: c for c in all_classes}
        for bl in blank_classes:
            map_dict.pop(bl, None)

    # Snapshot Serengeti Top 26
    elif mapping == "ss_26":
        # top 26 species
        ss_26 = ['wildebeest', 'zebra', 'gazelleThomsons', 'buffalo',
                 'hartebeest', 'elephant', 'human', 'impala',
                 'gazelleGrants', 'giraffe', 'warthog', 'guineaFowl',
                 'hyenaSpotted', 'otherBird', 'eland', 'lionFemale',
                 'hippopotamus', 'reedbuck', 'topi', 'baboon', 'dikDik',
                 'cheetah', 'secretaryBird', 'lionMale', 'serval',
                 'ostrich']
        # group others to Others category
        ss_other = [x for x in all_classes
                    if ((x not in ss_26) and (x not in blank_classes))]
        map_26 = {c: c for c in ss_26}
        map_other = {c: 'others' for c in ss_other}

        # combine top 26 and others category
        map_dict = {**map_26, **map_other}

    elif mapping == "ss_51":
        # remove blanks and low-occurence classes
        low_occurrence_classes = ['steenbok', 'cattle', 'bat']
        map_dict = {c: c for c in all_classes}
        for bl in blank_classes:
            map_dict.pop(bl, None)
        for bl in low_occurrence_classes:
            map_dict.pop(bl, None)

    elif mapping == 'ss_zebra_elephant':
        map_dict = {'elephant': 'elephant',
                    'zebra': 'zebra'}

    elif mapping == 'ee_nonblank_no_cannotidentify':
        map_dict = {c: c for c in all_classes}
        for bl in blank_classes:
            map_dict.pop(bl, None)
        map_dict.pop('CANNOTIDENTIFY', None)

    # Camera Catalogue: blank vs vehicle vs species
    elif mapping == 'cc_blank_vehicle_species':
        veh_bl = ['vehicle', 'blank']
        excl_classes = ['notblank', 'novehicle']
        map_spec = {c: 'species' for c in all_classes if c not in veh_bl}
        map_veh_bl = {c: c for c in veh_bl}
        map_dict = {**map_spec, **map_veh_bl}

        for lo in excl_classes:
            map_dict.pop(lo, None)

    # Camera Catalogue: species
    elif mapping == 'cc_species':
        veh_bl = ['vehicle', 'blank', 'notblank', 'novehicle']
        low_occurrence_classes = ['reptile', 'otter', 'pangolin', 'polecat',
                                  'MACAQUE', 'hyrax', 'roan', 'fire', 'sable']
        map_dict = {c: c for c in all_classes}
        for bl in veh_bl:
            map_dict.pop(bl, None)

        for lo in low_occurrence_classes:
            map_dict.pop(lo, None)

    elif mapping == 'sw_species':
        low_occurrence_classes = ['COUGAR', 'OTHERDOMESTIC', 'MARTEN',
                                  'PHEASANT',
                                  'HUMAN', 'MUSKRAT', 'BADGER',
                                  'REPTILESANDAMPHIBIANS', 'MOOSE',
                                  'WLVRN']

        map_dict = {c: c for c in all_classes}
        # remove blanks
        for bl in blank_classes:
            map_dict.pop(bl, None)

        # remove rare classes
        for lo in low_occurrence_classes:
            map_dict.pop(lo, None)

    else:
        NotImplementedError("Mapping %s not implemented" % mapping)

    return map_dict


def create_optimizer(name="standard"):
    """ Creates optimizers according to pre-defined options """
    if name == "sgd":
        opt = SGD(lr=0.01, momentum=0.9, decay=0)
    if name == "sgd_resnet":
        opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)
    elif name == "sgd_ss":
        opt = SGD(lr=0.01, momentum=0.9, decay=5e-4)
    elif name == "sgd_low":
        opt = SGD(lr=0.001, momentum=0.9, decay=0)
    elif name == "rmsprop":
        opt = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    elif name == "adagrad":
        opt = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    else:
        NotImplementedError("Optimizer %s not implemented" % name)
    return opt


# create data generators
def create_data_generators(cfg, target_shape, data_augmentation="none"):
    """ generate input data from a generator function that applies
    random / static transformations to the input """

    # handle color mode
    if target_shape[2] == 1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    # raw data generator required for calculating image statistics
    # on 2000 images
    logging.info("Initializing raw generator")
    datagen_raw = ImageDataGenerator(rescale=1./255)
    raw_generator = datagen_raw.flow_from_directory(
            cfg_path['images'] + 'train',
            target_size=target_shape[0:2],
            color_mode=color_mode,
            batch_size=2000,
            class_mode='sparse',
            seed=cfg_model['random_seed'])

    # no data augmentation
    if data_augmentation == "none":
        # data augmentation / preprocessing for train data
        datagen_train = ImageDataGenerator(
            rescale=1./255)
        # augmentation / preprocessing for test / validation data
        datagen_test = ImageDataGenerator(rescale=1./255)

    # Snapshot Serengeti mode
    elif data_augmentation == "ss":
        # data augmentation / preprocessing for train data
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            # samplewise_center=True,
            # samplewise_std_normalization=True,
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True,
            zoom_range=[0.9, 1])
        # augmentation / preprocessing for test / validation data
        datagen_test = ImageDataGenerator(
            rescale=1./255,
            featurewise_center=True,
            featurewise_std_normalization=True)

    # Not implemented data augmentation exception
    else:
        NotImplementedError("data_augmentation mode %s not implemented"
                            % data_augmentation)

    # create generators which serve images from directories for
    # test / train and validation data
    if cfg_model['image_iterator'] == 'double_iterator':
        batch_size = cfg['batch_size'] * 50
    else:
        batch_size = cfg['batch_size']

    logging.info("Initializing train generator")
    train_generator = datagen_train.flow_from_directory(
            cfg_path['images'] + 'train',
            target_size=target_shape[0:2],
            color_mode=color_mode,
            batch_size=batch_size,
            class_mode='sparse',
            seed=cfg_model['random_seed'])

    logging.info("Initializing test generator")
    test_generator = datagen_test.flow_from_directory(
            cfg_path['images'] + 'test',
            target_size=target_shape[0:2],
            color_mode=color_mode,
            batch_size=batch_size,
            class_mode='sparse',
            seed=cfg_model['random_seed'])

    logging.info("Initializing val generator")
    val_generator = datagen_test.flow_from_directory(
            cfg_path['images'] + 'val',
            target_size=target_shape[0:2],
            color_mode=color_mode,
            batch_size=batch_size,
            class_mode='sparse',
            seed=cfg_model['random_seed'])

    # fit data generators if required
    if any([datagen_train.featurewise_center,
            datagen_train.featurewise_std_normalization,
            datagen_train.zca_whitening]):

        # get random batch of raw training data
        x, y = raw_generator.next()

        # fit statistics from same batch of training data on the
        # data generators
        for gen in (datagen_train, datagen_test):
            gen.fit(x, seed=cfg_model['random_seed'])
            if any([gen.featurewise_center,
                    gen.featurewise_std_normalization]):
                logging.info("Featurewise center, means: %s" % gen.mean)
                logging.info("Featurewise center, std: %s" % gen.std)

    if cfg_model['image_iterator'] == 'double_iterator':
        res = ()
        for gen in (train_generator, test_generator, val_generator):
            big = DoubleIterator(gen, batch_size=cfg['batch_size'],
                                 seed=cfg_model['random_seed'])
            res = res + (big, )
        logging.info("Initialized DoubleIterator")
        return res

    # print all class mappings
    for gen, label in zip((train_generator, test_generator, val_generator),
                          ["train", "test", "val"]):
        logging.info("Class mapping for set: %s" % label)

        # empty class list
        classes_all = list()
        # mapping of index to class
        class_mapper = {v: k for k, v in gen.class_indices.items()}
        for i in range(0, len(class_mapper.keys())):
            classes_all.append(class_mapper[i])

        for k, v in gen.class_indices.items():
            logging.info("Class %s maps to index %s" % (k, v))

        logging.info("Full ordered mapping: %s" % classes_all)

    return train_generator, test_generator, val_generator


def create_callbacks(identifier='',
                     names=['checkpointer', 'checkpointer_best',
                            'csv_logger', 'tb_logger']):
    """ Create Callbacks for logging during training """
    # instantiate list of callbacks
    callbacks = []

    # handle identifier string
    if identifier is not '':
        identifier = identifier + '_'

    # add different callbacks if they are specified
    if 'checkpointer' in names:
        # save model weights after each epoch
        checkpointer = ModelCheckpoint(filepath=cfg_path['models'] +
                                       identifier +
                                       "model_{epoch:02d}_{val_loss:.2f}.hdf5",
                                       verbose=0,
                                       save_best_only=False)
        callbacks.append(checkpointer)

    if 'checkpointer_best' in names:
        # save best model with lowest test error
        checkpointer = ModelCheckpoint(filepath=cfg_path['models'] +
                                       identifier +
                                       "model_best.hdf5",
                                       verbose=0,
                                       save_best_only=True)
        callbacks.append(checkpointer)

    if 'csv_logger' in names:
        # log loss and accuracy to csv
        csv_logger = CSVLogger(cfg_path['logs'] + identifier +
                               'training.log')

        callbacks.append(csv_logger)

    if 'tb_logger' in names:
        # Tensorboard logger
        tb_logger = TensorBoard(log_dir=cfg_path['logs'], histogram_freq=0,
                                # batch_size=int(cfg['batch_size']),
                                write_graph=True
                                # write_grads=False, write_images=False,
                                # embeddings_freq=0,
                                # embeddings_layer_names=None,
                                # embeddings_metadata=None
                                )

        callbacks.append(tb_logger)

    if 'remote_logger' in names:
        # Remote Logger

        # get ip
        ip_call = run(['curl', 'checkip.amazonaws.com'], stdout=PIPE)
        ip = ip_call.stdout.decode("utf-8").strip("\n").replace('.', '-')
        ip_ec2 = 'http://ec2-' + ip + '.compute-1.amazonaws.com'

        rem_logger = RemoteMonitor(root=ip_ec2 + ':8080',
                                   path='/publish/epoch/end/',
                                   field='data',
                                   headers=None)
        # print / log server
        logging.info("Initializing Remote logger at: %s" % (ip_ec2 + ':8080'))
        print("Initializing Remote logger at: %s" % (ip_ec2 + ':8080'))
        callbacks.append(rem_logger)

    if 'log_disk' in names:
        logging_cb = LoggingCallback(logging=logging)
        callbacks.append(logging_cb)

    if 'early_stopping' in names:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=5, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if 'reduce_lr_on_plateau' in names:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, verbose=1,
                                      mode='auto', epsilon=0.0001,
                                      cooldown=0, min_lr=0.0001)
        callbacks.append(reduce_lr)

    if 'ss_learning_rate' in names:
        # learning rate adjustment scheme according to epoch number
        def lrnrate_dec(epoch):
            if epoch < 18:
                res = 0.01
            elif epoch < 29:
                res = 0.005
            elif epoch < 43:
                res = 0.001
            elif epoch < 52:
                res = 5e-4
            else:
                res = 1e-4

            logging.info("Setting learning rate to: %s" % res)
            return res

        # learning rate decay rule
        learning_rate_decay = LearningRateScheduler(lrnrate_dec)

        callbacks.append(learning_rate_decay)

    if 'ss_decay' in names:
        # learning rate adjustment scheme according to epoch number
        def decay_dec(epoch):
            if epoch < 18:
                res = 5e-4
            elif epoch < 29:
                res = 5e-4
            elif epoch < 43:
                res = 0
            elif epoch < 52:
                res = 0
            else:
                res = 0

            logging.info("Setting learning rate decay to: %s" % res)
            return res

        # learning rate decay rule
        learning_rate_decay = LRDecayScheduler(decay_dec)

        callbacks.append(learning_rate_decay)

    return callbacks


# Function to save model
def model_save(model, config=config, cfg_path=cfg_path,
               postfix=None, create_dir=True):

    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    # get path to save models
    path_to_save = cfg_path['save']

    # define model name to save
    model_id = config[project_id]['experiment_id']

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

    model.save(path_to_save + out_name + '.h5')


class LRDecayScheduler(Callback):
    """Learning rate decay scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate decay as output (float).
    """

    def __init__(self, schedule):
        super(LRDecayScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'decay'):
            raise ValueError('Optimizer must have a "decay" attribute.')
        decay = self.schedule(epoch) * 1.0
        if not isinstance(decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')

        K.set_value(self.model.optimizer.decay, decay)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, logging):
        Callback.__init__(self)
        self.logging = logging

    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join(
              "%s: %f" % (k, v) for k, v in logs.items()))
        self.logging.info(msg)
