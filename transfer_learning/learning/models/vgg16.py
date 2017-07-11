from keras.applications import VGG16


def build_model(num_classes, image_size=(224, 224, 3)):
    """ Define model architecture / parameters """
    # define model
    mod = VGG16(input_shape=image_size,
                classes=num_classes,
                weights=None)

    # save model data
    model_data = dict()
    model_data['model'] = mod
    model_data['input_shape'] = image_size

    return model_data
