import os
from config.config import config, cfg_path


# Function to save model
def model_save(model, config=config, cfg_path=cfg_path, postfix=None, create_dir=True):

    # extract project id for further loading project specifc configs
    project_id = config['projects']['panoptes_id']

    # get path to save models
    path_to_save = cfg_path['save']

    # define model name to save
    model_id = config[project_id]['identifier']

    path_to_save = path_to_save.replace("//", "/")

    if postfix is not None:
        out_name = '%s_%s' % (model_id, postfix)
    else:
        out_name = model_id

    # check path
    if not os.path.exists(path_to_save) & create_dir:
        os.mkdir(path_to_save)
    else:
        NameError("Path not Found")

    model.save(path_to_save + out_name + '.h5')



# function to load parameters used for model training
def model_param_loader(config=config):
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


    # build config dictionary for easier use in code
    cfg = dict()
    cfg['batch_size'] = batch_size
    cfg['num_classes'] = num_classes
    cfg['num_epochs'] = num_epochs
    cfg['data_augmentation'] = data_augmentation
    cfg['image_size_save'] = image_size_save
    cfg['image_size_model'] = image_size_model
    cfg['random_seed'] = int(config[project_id]['random_seed'])

    return cfg
