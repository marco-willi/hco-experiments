# model for handwritten-digits data on Zooniverse
from tools.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from config.config import config
from tools.model_helpers import model_save, model_param_loader


def train(train_dir, test_dir, val_dir):
    ##################################
    # Parameters
    ##################################

    cfg = model_param_loader(config)
    print_separator = "---------------------------------"

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

    train_generator = datagen_train.flow_from_urls(
            urls=train_dir.paths,
            labels=train_dir.labels,
            classes=['0', '1'],
            target_size=cfg['image_size_model'][0:2],
            batch_size=cfg['batch_size'],
            class_mode='binary')

    # this is a similar generator, for validation data
    test_generator = datagen_test.flow_from_urls(
            urls=test_dir.paths,
            labels=test_dir.labels,
            classes=['0', '1'],
            target_size=cfg['image_size_model'][0:2],
            batch_size=cfg['batch_size'],
            class_mode='binary')

    val_generator = datagen_test.flow_from_urls(
            urls=val_dir.paths,
            labels=val_dir.labels,
            classes=['0', '1'],
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

    model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_dir.paths) // cfg['batch_size'],
            epochs=cfg['num_epochs'])
#    ,
#            validation_data=test_generator,
#            validation_steps=len(test_dir.paths) // cfg['batch_size'])

    ##################################
    # Save
    ##################################

    model_save(model, config)



#
#import random as rand
#r = rand.sample([x for x in range(0,X_val.shape[0])],k=1)
#test_image = X_val[r,:,:,:]
#print("Predicted Class %s" % int(model.predict_classes(test_image)))
#array_to_img(test_image[0,:,:,:])
##################################
# Keras Data Pre-Processing
##################################

#
#def keras_preprocessing(X, Y, num_classes, target_shape=image_size_model):
#    if num_classes > 2:
#        Y = keras.utils.to_categorical(Y, num_classes)
#    # convert size if different
#    if not X.shape[1:4] == target_shape:
#        # create empty target array
#        new_size = tuple([X.shape[0]] + list(target_shape))
#        X_new = np.zeros(shape=new_size)
#        # loop over all images and resize
#        for i in range(0, X.shape[0]):
#            X_i = imresize(X[i, :, :, :],
#                           target_shape,
#                           interp='bilinear',
#                           mode=None)
#            X_new[i, :, :, :] = X_i
#        X = X_new
#    return X, Y