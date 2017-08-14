
############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_output_report <- "D:/Studium_GD/Zooniverse/Results/transfer_learning/reports/"
path_output_report <- "D:\\Studium_GD\\Zooniverse\\Results\\transfer_learning\\reports\\"

############################ -
# Parameters ----
############################ -


# Snapshot Serengeti - Top26 species
# project_id <- "ss"
# model <- "ss_species_26"
# project_name <- "Snapshot Serengeti"
# ts_id <- "201707271307"
# model_name <- "Species Top26"

# Snapshot Serengeti - Top51 species
# project_id <- "ss"
# model <- "ss_species_51"
# project_name <- "Snapshot Serengeti"
# ts_id <- "201708072308"
# model_name <- "Species Top51"


# Snapshot Serengeti - Blank vs Non Blank
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "ss"
# pred_file = "ss_blank_vs_non_blank_small_201707271407_preds_test"
# log_file <- "ss_blank_vs_non_blank_small_201707172207_training"
# model <- "blank_vs_non_blank_small"


# Elephant Expedition - blank vs non-blank
# project_id <- "elephant_expedition"
# ts_id <- "201708021608"
# model <- "ee_blank_vs_nonblank"
# subject_set <- "val_subject_set_ee_blank_vs_nonblank"
# project_name <- "Elephant Expedition"
# model_nam <- "Blank vs Non-Blank"


# Elephant Expedition - species
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_nonblank_201708030208_preds_test"
# log_file <- "ee_nonblank_201708021908_training"
# model <- "ee_nonblank"
# subject_set <- "test_subject_set_ee_nonblank"
# 

# Elephant Expedition - species no cannotidentify
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_nonblank_no_cannotidentify_201708050608_preds_val"
# log_file <- "ee_nonblank_no_cannotidentify_201708042308_training"
# model <- "ee_nonblank_no_cannotidentify"
# subject_set <- "val_subject_set_ee_nonblank_no_cannotidentify"

# Camera Catalogue - Blank vs Vehicle vs Species
project_id <- "camera_catalogue"
model <- "cc_blank_vehicle_species"
project_name <- "Camera Catalogue"
ts_id <- "201708052008"
model_name <- "Blank vs Vehicle vs Species"

# Camera Catalogue - Species
# project_id <- "camera_catalogue"
# model <- "cc_species"
# project_name <- "Camera Catalogue"
# ts_id <- "201708072308"
# model_name <- "Species"


# Elephant Expedition - blank vs non-blank
# project_id <- "elephant_expedition"
# model <- "ee_blank_vs_nonblank"
# ts_id <- "201708012008"
# project_name <- "Elephant Expedition"
# model_name <- "Blank vs Non-Blank"

# Elephant Expedition - Species
# project_id <- "elephant_expedition"
# model <- "ee_nonblank_no_cannotidentify"
# ts_id <- "201708042308"
# project_name <- "Elephant Expedition"
# model_name <- "Species (excl. Cannotidentify)"


# Snapshot Wisconsin - blank vs non-blank
# project_id <- "snapshot_wisconsin"
# model <- "sw_blank_vs_nonblank"
# ts_id <- "201708081608"
# project_name <- "Snapshot Wisconsin"
# model_name <- "Blank vs Non-Blank"

# Snapshot Wisconsin - species
# project_id <- "snapshot_wisconsin"
# model <- "sw_species"
# ts_id <- "201708092208"
# project_name <- "Snapshot Wisconsin"
# model_name <- "Species Detection"



############################ -
# Paths Project ----
############################ -

path_logs <- paste(path_main,"logs/",project_id,"/",sep="")
path_save <- paste(path_main,"save/",project_id,"/",sep="")
path_db <- paste(path_main,"db/",project_id,"/",sep="")
path_scratch <- paste(path_main,"scratch/",project_id,"/",sep="")
path_figures <- paste(path_main,"save/",project_id,"/figures/",sep="")


############################ -
# Load Functions & Data ----
############################ -

# Load Functions
source("analyses/plot_functions.R")

# Load data
source("analyses/load_data.R")

# save eval plots
# source("analyses/eval2.R")

# save subject set plots
# source("analyses/plot_subject_set.R")


############################ -
# Create Report ----
############################ -

library(knitr)
library(rmarkdown)

# render report
params = list(
  project_name=project_name,
  project_id=project_id,
  ts_id=ts_id,
  model_name=model_name,
  model_id=model,
  title=paste(project_name," - ",model_name,sep=""),
  author="Marco Willi",
  date=Sys.Date(),
  path_main=path_main
)

output <- "html"

rmarkdown::render(input="analyses/report_template.Rmd", 
                  params=params,
                  output_format = paste(output,"_document",sep=""),
                  output_file = paste(path_output_report,project_id,"_",model,".",output,sep=""))

