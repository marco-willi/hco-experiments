# model for handwritten-digits data on Zooniverse
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tools.data_generator import DataFetcher
import numpy as np
from main import test_dir, train_dir, val_dir, project_id
from keras.preprocessing.image import array_to_img
from config.config import config
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
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


##################################
# Data Generator
##################################

# generate junks of input data
data_fetcher_train = DataFetcher(train_dir, asynch_read=True,
                                 image_size=image_size_save[0:2],
                                 batch_size=eval(config[project_id]['batch_size_big']),
                                 disk_scratch = config['paths']['path_scratch'],
                                 random_shuffle_batches=True)

data_fetcher_test = DataFetcher(test_dir, asynch_read=True,
                                image_size=image_size_save[0:2],
                                n_big_batches=1,
                                disk_scratch = config['paths']['path_scratch'])

data_fetcher_val = DataFetcher(val_dir, asynch_read=True,
                               image_size=image_size_save[0:2],
                               n_big_batches=1,
                               disk_scratch = config['paths']['path_scratch'])

# generate input data from a generator function that applies
# random / static transformations to the input
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

##################################
# Model Definition
##################################

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=image_size_model))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


##################################
# Keras Data Pre-Processing
##################################

def keras_preprocessing(X, Y, num_classes, target_shape=image_size_model):
    if num_classes > 2:
        Y = keras.utils.to_categorical(Y, num_classes)
    X = X.astype('float32')
    X /= 255
    # convert size if different
    if not X.shape[1:4] == target_shape:
        if X.shape[4] == 1:
            raise ValueError("Image Transformation for 1-D images  \
                             not supported")
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

# get test data
X_test, Y_test = data_fetcher_test.nextBatch()
X_test, Y_test = keras_preprocessing(X_test, Y_test, num_classes=num_classes)

# get validation data
X_val, Y_val = data_fetcher_val.nextBatch()
X_val, Y_val = keras_preprocessing(X_val, Y_val, num_classes=num_classes)

# Training loop over number of epochs
for e in range(num_epochs):
    print("------------------------------------------")
    print("Epoch %d" % e)
    print("------------------------------------------")
    # loop over distinct batches of the training data
    # generated from the DataGenerator class
    for i in range(0, data_fetcher_train.n_batches):
        # get next batch
        X_train, Y_train = data_fetcher_train.nextBatch()
        # transform to keras specific formats
        X_train, Y_train = keras_preprocessing(X_train, Y_train,
                                               num_classes=num_classes)
        # data augmentation
        if data_augmentation:
            n_batches_used = 0
            # fit data generator on first batch
            if (e == 0) & (i == 0):
                datagen.fit(X_train)
                print("Finished fitting DataGenerator")
            # fit model with data from data generator
            model.fit_generator(
                    datagen.flow(X_train, Y_train,
                                 batch_size=batch_size,
                                 shuffle=False),
                    steps_per_epoch=(np.ceil(Y_train.shape[0] / batch_size)),
                    epochs=1,
                    validation_data=(X_test,Y_test))
        else:
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=1,
                      validation_data=(X_test,Y_test), verbose=1)


##################################
# Evaluation
##################################

model.test_on_batch(X_val, Y_val)
model.metrics_names


##################################
# Save
##################################

path_to_save = config['paths']['path_final_models']
model_id = config['model']['identifier']
model.save(path_to_save + model_id + '.h5')



#
import random as rand
r = rand.sample([x for x in range(0,X_val.shape[0])],k=1)
test_image = X_val[r,:,:,:]
print("Predicted Class %s" % int(model.predict_classes(test_image)))
array_to_img(test_image[0,:,:,:])
