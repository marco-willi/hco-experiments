""" Predict Images """
from tools.predictor_external import PredictorExternal
from config.config import cfg_path
from keras.preprocessing.images import ImageDataGenerator


def predict_images():
    """ Predict Images in Directory """
    # prepare Keras Data-Generator for fitting on Keras
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True)

    # class mapping according to model
    class_list_species = [
                  'aardvark', 'aardwolf', 'baboon', 'batEaredFox', 'buffalo',
                  'bushbuck', 'caracal', 'cheetah', 'civet', 'dikDik',
                  'duiker', 'eland', 'elephant', 'gazelleGrants',
                  'gazelleThomsons', 'genet', 'giraffe', 'guineaFowl', 'hare',
                  'hartebeest', 'hippopotamus', 'honeyBadger', 'human',
                  'hyenaSpotted', 'hyenaStriped', 'impala', 'insectSpider',
                  'jackal', 'koriBustard', 'leopard', 'lionFemale', 'lionMale',
                  'mongoose', 'ostrich', 'otherBird', 'porcupine', 'reedbuck',
                  'reptiles', 'rhinoceros', 'rodents', 'secretaryBird',
                  'serval', 'topi', 'vervetMonkey', 'vulture', 'warthog',
                  'waterbuck', 'wildcat', 'wildebeest', 'zebra', 'zorilla']

    class_list_blank = ['blank', 'non_blank']

    # score images
    model_files = ['/host/data_hdd/save/ss/ss_species_51_201708072308.hdf5',
                   '/host/data_hdd/models/ss/' +
                   'ss_blank_vs_non_blank_small_201707172207_model_best.hdf5']

    # class mapping
    class_lists = [class_list_species, class_list_blank]

    # input images for scoring
    pred_path = cfg_path['images']

    # save csv with scored images
    output_path = cfg_path['save']
    out_file_names = ['predictions_species.csv', 'predictions_blank.csv']

    for model_file, class_list, output_file_name in \
     zip(model_files, class_lists, out_file_names):

        # predictor
        predictor = PredictorExternal(
            path_to_model=model_file,
            keras_datagen=datagen,
            class_list=class_list)

        # predict images in path
        predictor.predict_path(path=pred_path, output_path=output_path,
                               output_file_name=output_file_name)


if __name__ == "__main__":
    predict_images()
