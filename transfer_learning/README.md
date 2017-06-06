# Transfer Learning

## Introduction

The goal of this project is to train machine classifiers for the identification of animal species in camera trap images. The idea is to combine the data from multiple projects to obtain better individual classifiers. The code is aimed at being run on a AWS GPU instance, while fetching data from the Zooniverse database, and should be general enough to being useful for any image classification tasks in the Zooniverse universe.

## Run code

To run a model one needs:
* a Zooniverse account
* access to a Zooniverse project (collaborator role or project owner)
* a project with image data for classification tasks
* a way to generate labels for the images
* (ideally) access to a GPU server

## Code Structure

### aws_build
Code to set up AWS GPU instance. 

### config
Configuration file: models, paths and credentials.

### models
Model files for different projects.

### sandbox
Files not relevant to running the main program.

### tools
Different helper functions, e.g. fetching data.
