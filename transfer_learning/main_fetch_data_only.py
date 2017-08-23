# load modules
from config.config import config, cfg_path, cfg_model
from tools.project import Project

# Main Program
def main():
    #######################
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


if __name__ == "__main__":
        main()
