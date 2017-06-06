# Function to save model
def model_save(model, config, postfix=None):
    path_persistent = config['paths']['path_persistent']
    path_to_save = path_persistent + config['paths']['path_final_models']
    model_id = config['model']['identifier']

    path_to_save = path_to_save.replace("//", "/")

    if postfix is not None:
        out_name = '%s_%s' % (model_id, postfix)
    else:
        out_name = model_id

    model.save(path_to_save + out_name + '.h5')


# function to load parameters used for model training
def model_param_loader(config):
    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    batch_size = eval(config[project_id]['batch_size'])
    num_classes = int(config[project_id]['num_classes'])
    num_epochs = int(config[project_id]['num_epochs'])
    data_augmentation = eval(config[project_id]['data_augmentation'])

    image_size_save = config[project_id]['image_size_save'].split(',')
    image_size_save = tuple([int(x) for x in image_size_save])

    image_size_model = config[project_id]['image_size_model'].split(',')
    image_size_model = tuple([int(x) for x in image_size_model])

    scratch = config['paths']['path_scratch']
    path_persistent = config['paths']['path_persistent']

    # build config dictionary for easier use in code
    cfg = dict()
    cfg['batch_size'] = batch_size
    cfg['num_classes'] = num_classes
    cfg['num_epochs'] = num_epochs
    cfg['data_augmentation'] = data_augmentation
    cfg['image_size_save'] = image_size_save
    cfg['image_size_model'] = image_size_model
    cfg['scratch'] = scratch
    cfg['persistent'] = path_persistent

    return cfg
