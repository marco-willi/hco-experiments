from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

def center_crop(x, center_crop_size=(330, 330), **kwargs):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    cropped = x[:, centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh]
    return cropped

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest',
        rotation_range=5,
        #width_shift_range=0.05,
        #height_shift_range=0.2,
        #shear_range=0.2,
        zoom_range=[0.9, 1],
        horizontal_flip=True)


save_dir = 'D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\scratch\\augmentation_test'
img_path = "D:\\Studium_GD\\Zooniverse\\Data\\transfer_learning_project\\images\\camera_catalogue\\all\\hippopotamus\\4474370_0.jpeg"
img_path = "D:\\Studium_GD\\Zooniverse\\Data\\images_from_camera_traps\\elephant_expedition_sample_image.jpeg"
img_path = "D:\\Studium_GD\\Zooniverse\\Data\\images_from_camera_traps\\ss_med.JPG"
img = load_img(img_path)  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_dir, save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely





from keras.preprocessing.image import ImageDataGenerator,standardize,random_transform
# input generator with standardization on
datagenX = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    featurewise_standardize_axis=(0, 2, 3),
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect',
    seed=0,
    verbose=1)

# output generator with standardization off
datagenY = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect',
    seed=0)

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

datagenX.config['random_crop_size'] = (800, 800)
datagenY.config['random_crop_size'] = (800, 800)
datagenX.config['center_crop_size'] = (512, 512)
datagenY.config['center_crop_size'] = (360, 360)

# customize the pipeline
datagenX.set_pipeline([random_crop, random_transform, standardize, center_crop])
datagenY.set_pipeline([random_crop, random_transform, center_crop])

# flow from directory is extended to support more format and also you can even use your own reader function
# here is an example of reading image data saved in csv file
# datagenX.flow_from_directory(csvFolder, image_reader=csvReaderGenerator, read_formats={'csv'}, reader_config={'target_size':(572,572),'resolution':20, 'crange':(0,100)}, class_mode=None, batch_size=1)

dgdx= datagenX.flow(x, class_mode=None, read_formats={'png'}, batch_size=2)
dgdy= datagenY.flow(x,  class_mode=None, read_formats={'png'}, batch_size=2)

# you can now fit a generator as well
datagenX.fit_generator(dgdx, nb_iter=100)

# here we sychronize two generator and combine it into one
train_generator = dgdx+dgdy

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
