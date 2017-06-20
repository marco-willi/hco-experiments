from tools.image import ImageDataGenerator
from config.config import config, cfg_path
import os


# create class mappings
def create_class_mappings(mapping="1_on_1"):
    # all classes
    cfg = model_param_loader(config)

    # default 1 to 1 mapping
    if mapping == "1_on_1":
        map_dict = {c:c for c in cfg['classes']}

    # SS blank vs empty
    if mapping == "ss_blank_vs_nonblank":
        map_dict = dict()
        for c in cfg['classes']:
            if c == 'blank':
                map_dict[c] = 'blank'
            else:
                map_dict[c] = 'non_blank'

    if mapping == "nonblank":
        map_dict = {c:c for c in cfg['classes']}
        map_dict.pop('blank', None)



    return map_dict


# create data generators
def create_data_generators(data_augmentation="none"):
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


# Function to save model
def model_save(model, config=config, cfg_path=cfg_path, postfix=None, create_dir=True):

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

# function to load parameters used for model training
def model_param_loader(config=config):
    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    batch_size = eval(config[project_id]['batch_size'])
    num_classes = int(config[project_id]['num_classes'])
    num_epochs = int(config[project_id]['num_epochs'])
    data_augmentation = config[project_id]['data_augmentation']

    image_size_save = config[project_id]['image_size_save'].split(',')
    image_size_save = tuple([int(x) for x in image_size_save])

    image_size_model = config[project_id]['image_size_model'].split(',')
    image_size_model = tuple([int(x) for x in image_size_model])

    classes = config[project_id]['classes'].replace("\n","").split(",")


    # build config dictionary for easier use in code
    cfg = dict()
    cfg['batch_size'] = batch_size
    cfg['num_classes'] = num_classes
    cfg['num_epochs'] = num_epochs
    cfg['data_augmentation'] = data_augmentation
    cfg['image_size_save'] = image_size_save
    cfg['image_size_model'] = image_size_model
    cfg['random_seed'] = int(config[project_id]['random_seed'])
    cfg['classes'] = classes

    return cfg


