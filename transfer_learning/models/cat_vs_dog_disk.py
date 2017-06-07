# model for handwritten-digits data on Zooniverse
from tools.image import ImageDataGenerator
from tools.image_url_loader import ImageUrlLoader
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from config.config import config
from tools.model_helpers import model_save, model_param_loader
import time


def train(train_dir, test_dir, val_dir):
    ##################################
    # Parameters
    ##################################

    cfg = model_param_loader(config)
    print_separator = "---------------------------------"

    ##################################
    # Save data on disk
    ##################################

    # define loaders
    train_data_loader = ImageUrlLoader()
    test_data_loader = ImageUrlLoader()
    val_data_loader = ImageUrlLoader()

    # save to disk
    print(print_separator)
    print("Saving data on disk ....")
    print(print_separator)
    print("Saving train data ....")
    time_s = time.time()
    train_data_loader.storeOnDisk(urls=train_dir.paths,
                                  labels=train_dir.labels,
                                  ids=train_dir.unique_ids,
                                  path=cfg['scratch'] + 'train',
                                  target_size=cfg['image_size_save'][0:2],
                                  chunk_size=100)
    print("Saving test data ....")
    test_data_loader.storeOnDisk(urls=test_dir.paths,
                                 labels=test_dir.labels,
                                 ids=test_dir.unique_ids,
                                 path=cfg['scratch'] + 'test',
                                 target_size=cfg['image_size_save'][0:2],
                                 chunk_size=100)

    print("Saving val data ....")
    val_data_loader.storeOnDisk(urls=val_dir.paths,
                                labels=val_dir.labels,
                                ids=val_dir.unique_ids,
                                path=cfg['scratch'] + 'val',
                                target_size=cfg['image_size_save'][0:2],
                                chunk_size=100)

    print("Finished saving on disk after %s minutes" %
          ((time.time() - time_s) // 60))


    ##################################
    # Data Generator
    ##################################

    # generate input data from a generator function that applies
    # random / static transformations to the input
    datagen_train = ImageDataGenerator(
        rescale=1./255,  # rescale input values
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    datagen_test = ImageDataGenerator(
        rescale=1./255)

    train_generator = datagen_train.flow_from_directory(
            cfg['scratch'] + 'train',
            target_size=cfg['image_size_model'][0:2],  # all images will be resized
            batch_size=cfg['batch_size'],
            class_mode='binary')

    # this is a similar generator, for validation data
    test_generator = datagen_test.flow_from_directory(
            cfg['scratch'] + 'test',
            target_size=cfg['image_size_model'][0:2],
            batch_size=cfg['batch_size'],
            class_mode='binary')

    val_generator = datagen_test.flow_from_directory(
            cfg['scratch'] + 'val',
            target_size=cfg['image_size_model'][0:2],
            batch_size=cfg['batch_size'],
            class_mode='binary')

    ##################################
    # Model Definition
    ##################################

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=cfg['image_size_model']))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    ##################################
    # Training
    ##################################

    time_s = time.time()
    model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_dir.paths) // cfg['batch_size'],
            epochs=cfg['num_epochs'],
            validation_data=test_generator,
            validation_steps=len(test_dir.paths) // cfg['batch_size'],
            workers=4,
            pickle_safe=True)

    print("Finished training after %s minutes" %
          ((time.time() - time_s) // 60))

    ##################################
    # Evaluation
    ##################################

    model.evaluate_generator(
            test_generator,
            steps=len(test_dir.paths) // cfg['batch_size'],
            workers=4,
            pickle_safe=True)

    ##################################
    # Save
    ##################################

    model_save(model, config)
