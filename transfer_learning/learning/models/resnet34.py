from learning.keras_resnet.resnet import ResnetBuilder


def build_model(num_classes, image_size=(224, 224, 3)):
    """ Define model architecture / parameters """
    # define model
    model = ResnetBuilder.build_resnet_34(image_size, num_classes)

    # save model data
    model_data = dict()
    model_data['model'] = model
    model_data['input_shape'] = image_size

    return model_data
