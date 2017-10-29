# load modules
from config.config import config, cfg_path, cfg_model, logging
from tools.project import Project
from tools.experiment import Experiment
from learning.model_components import create_class_mappings


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
                     random_state=cfg_model['random_seed'],
                     max_labels_per_subject=1)

    # create separate directories with image data for this experiment
    # use only links to original images to save space
    exp.createExpDataSet(link_only=bool(eval(config['general']['link_only'])),
                         clear_old_files=False,
                         splits=cfg_model['experiment_data'],
                         split_mode=cfg_model['split_mode'])

    logging.info("Finished")


if __name__ == "__main__":
        main()
