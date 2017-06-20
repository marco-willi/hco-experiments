# load modules
from config.config import config, cfg_path
from tools.project import Project, Experiment


# Main Program
def main():

    ########################
    # Define a Project
    ########################

    # project id
    project_id = config['projects']['panoptes_id']

    # project classes
    project_classes = config[project_id]['classes'].replace("\n", "").split(",")

    # create Project boject
    project = Project(name=config[project_id]['identifier'],
                      panoptes_id=int(project_id),
                      classes=project_classes,
                      cfg_path=cfg_path,
                      config=config)

    # create Subject Sets
    project.createSubjectSet(mode=config[project_id]['subject_mode'])

    # save all subjects on Disk
    project.saveSubjectSetOnDisk()

    ########################
    # Define Experiment
    ########################

    exp = Experiment(name="mnist", project=project,
                     class_list=project_classes,
                     train_size=0.9)

    exp.createExpDataSet(link_only=False)

    ########################
    # Call Model
    ########################

    exp.addModelFile(config[project_id]['model_file'])

    exp.train()


if __name__ == "__main__":
        main()






