# Transfer Learning

## Introduction

The goal of this project is to train machine classifiers for the identification of animal species in camera trap images. The idea is to combine the data from multiple projects to obtain better individual classifiers. The code is aimed at being run on a AWS GPU instance, while fetching data from the Zooniverse database, and should be general enough to being useful for any image classification tasks in the Zooniverse universe.

## Run code

To run a model one needs:
* a Zooniverse account and access to a Zooniverse project (collaborator role or project owner)
  or Zooniverse data dumps for a project (subject and classification data)
* a project with image data for classification tasks
* a way to generate labels for the images
* (ideally) access to a GPU server

## Code Structure

### aws_build
Code to set up AWS GPU instance. 

### config
Configuration file: models, paths and credentials.

### learning
File to run the training process. Heavily relies on Keras. Also contains different models.

### sandbox
Files not relevant to running the main program.

### tools
Different helper functions, e.g. fetching data, classes for projects, experiments and subjects.

### db
Functions to deal with different raw data formats. The functions should be able to transform raw data into subject and subject set classes as defined in the tools module in order to be processed by the main program.
