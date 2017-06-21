# load modules
from config.config import config, cfg_path
from tools.project import Project
from tools.model import Model
from tools.experiment import Experiment
from learning.helpers import model_param_loader, create_class_mappings


# Main Program
def main():

    ########################
    # Define a Project
    ########################

    # project id
    project_id = config['projects']['panoptes_id']

    cfg = model_param_loader(config)

    # project classes
    project_classes = cfg['classes']

    # create Project object
    project = Project(name=cfg['identifier'],
                      panoptes_id=int(project_id),
                      classes=project_classes,
                      cfg_path=cfg_path,
                      config=config)

    # create Subject Sets
    project.createSubjectSet(mode=cfg['subject_mode'])

    # save all subjects on Disk
    project.saveSubjectSetOnDisk()

    ########################
    # Define Experiment
    ########################

    # map classes, defined in create_class_mappings function
    class_mapper = create_class_mappings(cfg['class_mapping'])

    # experiment object
    exp = Experiment(name="mnist", project=project,
                     class_mapper=class_mapper,
                     train_size=0.9)

    # create separate directories with image data for this experiment
    # use only links to save space
    exp.createExpDataSet(link_only=False)

    ########################
    # Define Model
    ########################

    # create model object
    model = Model(train_set=exp.train_set,
                  test_set=exp.test_set,
                  val_set=exp.val_set,
                  mod_file=cfg['model_file'],
                  pre_processing=cfg['pre_processing'],
                  config=config,
                  cfg_path=cfg_path,
                  callbacks=cfg['callbacks'],
                  optimizer=cfg['optimizer'],
                  num_classes=len(class_mapper.keys()))

    ########################
    # Train Model
    ########################

    exp.addModel(model)

    exp.train()


if __name__ == "__main__":
        main()






