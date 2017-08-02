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
    project = Project(name=str(project_id),
                      panoptes_id=project_id,
                      classes=project_classes,
                      cfg_path=cfg_path,
                      config=config)

    # create Subject Sets
    project.createSubjectSet(mode=cfg_model['subject_mode'])

    # save all subjects / images on disk that are not already there
    project.saveSubjectSetOnDisk(overwrite=False)

    ########################
    # Define Experiment
    ########################

    # map classes as specified by the current experiment,
    # choose from a mapping defined in create_class_mappings() function
    class_mapper = create_class_mappings(cfg_model['class_mapping'])

    # experiment object
    exp = Experiment(name=cfg_model['experiment_id'], project=project,
                     class_mapper=class_mapper,
                     train_size=cfg_model['train_size'],
                     test_size=cfg_model['test_size'],
                     equal_class_sizes=bool(cfg_model['balanced_classes']),
                     random_state=cfg_model['random_seed'])

    # create separate directories with image data for this experiment
    # use only links to original images to save space
    exp.createExpDataSet(link_only=bool(eval(config['general']['link_only'])),
                         clear_old_files=False,
                         splits=cfg_model['experiment_data'])

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
                  num_classes=len(set(class_mapper.values())))

    ########################
    # Train Model
    ########################

    # add model to experiment
    exp.addModel(model)

    # prepare / initialize model
    exp.prep_model()

    # train model
    exp.train()

    ########################
    # Evaluate Model
    ########################

    exp.evaluate()

    ########################
    # Save some information
    ########################

    save_objects = {'cfg_model': cfg_model,
                    'train_set': exp.train_set,
                    'test_set': exp.test_set,
                    'val_set': exp.val_set}

    pickle.dump(save_objects, open(cfg_path['save'] +
                                   cfg_model['experiment_id'] +
                                   '_' + str(cfg_model['ts']) +
                                   '_objects.pkl', "wb"))


if __name__ == "__main__":
        main()
