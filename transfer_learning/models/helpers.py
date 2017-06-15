from tools.image import ImageDataGenerator


# create data generators
def create_data_generators(cfg, cfg_path, data_augmentation=False):
    # generate input data from a generator function that applies
    # random / static transformations to the input

    # data augmentation
    if data_augmentation:
        datagen_train = ImageDataGenerator(
            rescale=1./255,  # rescale input values
            horizontal_flip=False)
    else:
        datagen_train = ImageDataGenerator(rescale=1./255)

    datagen_test = ImageDataGenerator(rescale=1./255)

    # create generators
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