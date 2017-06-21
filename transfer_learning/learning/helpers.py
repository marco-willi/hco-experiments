from tools.image import ImageDataGenerator
from config.config import config, cfg_path, cfg_model
import os
from keras.optimizers import rmsprop, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.callbacks import LearningRateScheduler

# create class mappings
def create_class_mappings(mapping="1_on_1", excl_classes=None,
                          only_these_classes=None):
    # all classes
    cfg = cfg_model
    all_classes = cfg['classes']

    if excl_classes is not None:
        all_classes = [x for x in all_classes if x not in excl_classes]

    if only_these_classes is not None:
        all_classes = [x for x in all_classes if x in only_these_classes]

    # default 1 to 1 mapping
    if mapping == "1_on_1":
        map_dict = {c: c for c in all_classes}

    # blank vs non-blank
    if mapping == "blank_vs_nonblank":
        map_dict = dict()
        for c in all_classes:
            if c == 'blank':
                map_dict[c] = 'blank'
            else:
                map_dict[c] = 'non_blank'

    # non blanks
    if mapping == "nonblank":
        map_dict = {c: c for c in all_classes}
        map_dict.pop('blank', None)

    return map_dict

def create_optimizer(name="standard"):
    if name == "standard":
        opt = SGD(lr=0.0001, decay=0)
    else:
        IOError("Optimizer %s not implemented" % name)
    return opt


# create data generators
def create_data_generators(cfg, data_augmentation="none"):
    """ generate input data from a generator function that applies
    random / static transformations to the input """

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
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
            zoom_range=[0.9, 1])
        # augmentation / preprocessing for test / validation data
        datagen_test = ImageDataGenerator(
            rescale=1./255,
            samplewise_center=True,
            samplewise_std_normalization=True)

    # Not implemented data augmentation exception
    else:
        IOError("data_augmentation mode %s not implemented"
                % data_augmentation)

    # create generators which serve images from directories for
    # test / train and validation data
    if cfg['image_size_model'][2] == 1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    train_generator = datagen_train.flow_from_directory(
            cfg_path['images'] + 'train',
            target_size=cfg['image_size_model'][0:2],
            color_mode=color_mode,
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    test_generator = datagen_test.flow_from_directory(
            cfg_path['images'] + 'test',
            target_size=cfg['image_size_model'][0:2],
            color_mode=color_mode,
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    val_generator = datagen_test.flow_from_directory(
            cfg_path['images'] + 'val',
            target_size=cfg['image_size_model'][0:2],
            color_mode=color_mode,
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    return train_generator, test_generator, val_generator


def create_callbacks(names=['checkpointer', 'csv_logger', 'tb_logger']):
    # list of callbacks
    callbacks = []

    if 'checkpointer' in names:
        # save model weights after each epoch if training loss decreases
        checkpointer = ModelCheckpoint(filepath=cfg_path['models'] +
                                       "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                       verbose=1,
                                       save_best_only=True)
        callbacks.append(checkpointer)

    if 'csv_logger' in names:
        # log to csv
        csv_logger = CSVLogger(cfg_path['logs'] + 'training.log')

        callbacks.append(csv_logger)

    if 'tb_logger' in names:
        # Tensorboard logger
        tb_logger = TensorBoard(log_dir=cfg_path['logs'], histogram_freq=0,
                                #batch_size=int(cfg['batch_size']),
                                write_graph=True
                                #write_grads=False, write_images=False,
                                #embeddings_freq=0,
                                #embeddings_layer_names=None,
                                #embeddings_metadata=None
                                )

        callbacks.append(tb_logger)

    if 'ss_learning_rate':
        # learning rate function
        def lrnrate_dec(epoch):
            if epoch < 18:
                return 0.01
            elif epoch < 29:
                return 0.005
            elif epoch < 43:
                return 0.001
            elif epoch < 52:
                return 5e-4
            else:
                return 1e-4

        # learning rate decay rule
        learning_rate_decay = LearningRateScheduler(lrnrate_dec)

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
    model_id = config[project_id]['identifier']

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
