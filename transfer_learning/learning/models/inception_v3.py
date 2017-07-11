from keras.applications import InceptionV3


def build_model(num_classes, image_size=(299, 299, 3)):
    mod = InceptionV3(input_shape=image_size,
                      classes=num_classes,
                      weights=None)

    # save model data
    model_data = dict()
    model_data['model'] = mod
    model_data['input_shape'] = image_size

    return model_data
