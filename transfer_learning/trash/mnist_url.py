# model for handwritten-digits data on Zooniverse
from tools.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from config.config import config
from tools.model_helpers import model_save, model_param_loader
import time


def train(train_set, test_set, val_set):
    ##################################
    # Parameters
    ##################################

    cfg = model_param_loader(config)

    ##################################
    # Data Generator
    ##################################

    # generate input data from a generator function that applies
    # random / static transformations to the input
    datagen_train = ImageDataGenerator(
        rescale=1./255,  # rescale input values
        horizontal_flip=False)

    datagen_test = ImageDataGenerator(
        rescale=1./255)

    urls, labels = train_set.getAllURLsLabels()
    train_generator = datagen_train.flow_from_urls(
            urls=urls,
            labels=labels,
            target_size=cfg['image_size_model'][0:2],
            color_mode='grayscale',
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    # this is a similar generator, for validation data
    urls, labels = test_set.getAllURLsLabels()
    test_generator = datagen_test.flow_from_urls(
            urls=urls,
            labels=labels,
            target_size=cfg['image_size_model'][0:2],
            color_mode='grayscale',
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    urls, labels = val_set.getAllURLsLabels()
    val_generator = datagen_test.flow_from_urls(
            urls=urls,
            labels=labels,
            target_size=cfg['image_size_model'][0:2],
            color_mode='grayscale',
            batch_size=cfg['batch_size'],
            class_mode='sparse')

    ##################################
    # Model Definition
    ##################################

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=cfg['image_size_model']))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(cfg['num_classes'], activation='softmax'))

    # initiate RMSprop optimizer
    opt = rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    ##################################
    # Training
    ##################################

    time_s = time.time()
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n // cfg['batch_size'],
            epochs=cfg['num_epochs'],
            workers=4,
            pickle_safe=False,
            validation_data=test_generator,
            validation_steps=test_generator.n // cfg['batch_size'])

    print("Finished training after %s minutes" %
          ((time.time() - time_s) // 60))

    ##################################
    # Evaluation
    ##################################

    # Test Data
    eval_metrics = model.evaluate_generator(
                    test_generator,
                    steps=test_generator.n // cfg['batch_size'],
                    workers=4,
                    pickle_safe=False)

    # print evaluation
    print("Test Results")
    for name, value in zip(model.metrics_names, eval_metrics):
        print("%s: %s" % (name, value))

    # Validation Data
    eval_metrics = model.evaluate_generator(
                    val_generator,
                    steps=val_generator.n // cfg['batch_size'],
                    workers=4,
                    pickle_safe=False)

    # print evaluation
    print("Validation Results")
    for name, value in zip(model.metrics_names, eval_metrics):
        print("%s: %s" % (name, value))

    ##################################
    # Save
    ##################################

    model_save(model, config)
