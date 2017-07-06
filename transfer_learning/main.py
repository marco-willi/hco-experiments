# load modules
from config.config import config, cfg_path, cfg_model
from tools.project import Project
from tools.experiment import Experiment
from tools.model import Model
from learning.helpers import create_class_mappings
import dill as pickle


# Main Program
def main():

    ########################
    # Define a Project
    ########################

    # project id
    project_id = config['projects']['panoptes_id']

    # project classes
    project_classes = cfg_model['classes']

    # create Project object
    project = Project(name=cfg_model['identifier'],
                      panoptes_id=project_id,
                      classes=project_classes,
                      cfg_path=cfg_path,
                      config=config)

    # create Subject Sets
    project.createSubjectSet(mode=cfg_model['subject_mode'])

    # save all subjects / images on disk that are not already there
    project.saveSubjectSetOnDisk()

    ########################
    # Define Experiment
    ########################

    # map classes, defined in create_class_mappings function
    class_mapper = create_class_mappings(cfg_model['class_mapping'])

    # experiment object
    exp = Experiment(name="mnist", project=project,
                     class_mapper=class_mapper,
                     train_size=0.9)

    # create separate directories with image data for this experiment
    # use only links to save space
    exp.createExpDataSet(link_only=bool(eval(config['general']['link_only'])))

    ########################
    # Define Model
    ########################

    # create model object
    model = Model(train_set=exp.train_set,
                  test_set=exp.test_set,
                  val_set=exp.val_set,
                  mod_file=cfg_model['model_file'],
                  pre_processing=cfg_model['pre_processing'],
                  config=config,
                  cfg_path=cfg_path,
                  callbacks=cfg_model['callbacks'],
                  optimizer=cfg_model['optimizer'],
                  num_classes=len(class_mapper.keys()))

    ########################
    # Train Model
    ########################

    exp.addModel(model)

    exp.train()

    ########################
    # Save some information
    ########################

    save_objects = {'cfg_model': cfg_model,
                    'train_set': exp.train_set,
                    'test_set': exp.test_set,
                    'val_set': exp.val_set}

    pickle.dump(save_objects, open(cfg_path['save'] + 'objects.pkl', "wb"))


if __name__ == "__main__":
        main()
