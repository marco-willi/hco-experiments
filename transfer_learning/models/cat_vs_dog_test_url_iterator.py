# model for handwritten-digits data on Zooniverse
import keras
#from keras.preprocessing.image import ImageDataGenerator
from tools.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tools.data_generator import DataFetcher
import numpy as np
from main import test_dir, train_dir, val_dir, project_id
from keras.preprocessing.image import array_to_img
from config.config import config
from scipy.misc import imresize

##################################
# Parameters
##################################

batch_size = eval(config[project_id]['batch_size'])
num_classes = int(config[project_id]['num_classes'])
num_epochs = int(config[project_id]['num_epochs'])
data_augmentation = eval(config[project_id]['data_augmentation'])

image_size_save = config[project_id]['image_size_save'].split(',')
image_size_save = tuple([int(x) for x in image_size_save])

image_size_model = config[project_id]['image_size_model'].split(',')
image_size_model = tuple([int(x) for x in image_size_model])

print_separator = "------------------------------------------"

##################################
# Data Generator
##################################

# generate input data from a generator function that applies
# random / static transformations to the input
train_datagen = ImageDataGenerator(
    rescale=1./255, # rescale input values
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally
                          # (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically
                           # (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)    # randomly flip images


test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_urls(
        urls=train_dir.paths,
        labels=train_dir.labels,
        classes=['0','1'],
        target_size=image_size_model[0:2],
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_urls(
        urls=test_dir.paths,
        labels=test_dir.labels,
        classes=['0','1'],
        target_size=image_size_model[0:2],
        batch_size=batch_size,
        class_mode='binary')

val_generator = test_datagen.flow_from_urls(
        urls=val_dir.paths,
        labels=val_dir.labels,
        classes=['0','1'],
        target_size=image_size_model[0:2],
        batch_size=batch_size,
        class_mode='binary')


##################################
# Model Definition
##################################


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=image_size_model))
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
# Keras Data Pre-Processing
##################################


def keras_preprocessing(X, Y, num_classes, target_shape=image_size_model):
    if num_classes > 2:
        Y = keras.utils.to_categorical(Y, num_classes)
    # convert size if different
    if not X.shape[1:4] == target_shape:
        # create empty target array
        new_size = tuple([X.shape[0]] + list(target_shape))
        X_new = np.zeros(shape=new_size)
        # loop over all images and resize
        for i in range(0, X.shape[0]):
            X_i = imresize(X[i, :, :, :],
                           target_shape,
                           interp='bilinear',
                           mode=None)
            X_new[i, :, :, :] = X_i
        X = X_new
    return X, Y

##################################
# Training
##################################

model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_dir.unique_ids) // batch_size,
        epochs=5,
        validation_data=test_generator,
        validation_steps=10)


##################################
# Evaluation
##################################

##################################
# Save
##################################

path_to_save = config['paths']['path_final_models']
model_id = config['model']['identifier']
model.save(path_to_save + model_id + '.h5')



#
#import random as rand
#r = rand.sample([x for x in range(0,X_val.shape[0])],k=1)
#test_image = X_val[r,:,:,:]
#print("Predicted Class %s" % int(model.predict_classes(test_image)))
#array_to_img(test_image[0,:,:,:])
