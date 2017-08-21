
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


# Camera Catalogue - Blank vs Vehicle vs Species
# project_id <- "camera_catalogue"
# model <- "cc_blank_vehicle_species"
# project_name <- "Camera Catalogue"
# ts_id <- "201708052008"
# model_name <- "Blank vs Vehicle vs Species"

# Camera Catalogue - Blank vs Vehicle vs Species (no dups)
# project_id <- "camera_catalogue"
# model <- "cc_blank_vehicle_species_v2"
# project_name <- "Camera Catalogue"
# ts_id <- "201708200608"
# model_name <- "Blank vs Vehicle vs Species"

# Camera Catalogue - Species OLD
# project_id <- "camera_catalogue"
# model <- "cc_species"
# project_name <- "Camera Catalogue"
# ts_id <- "201708072308"
# model_name <- "Species"

# Camera Catalogue - Species NEW
project_id <- "camera_catalogue"
model <- "cc_species_v2"
project_name <- "Camera Catalogue"
ts_id <- "201708210308"
model_name <- "Species"

# Camera Catalogue - Species Fine Tune SS 51 all
# project_id <- "camera_catalogue"
# model <- "cc_species_ss51_finetune_all"
# project_name <- "Camera Catalogue"
# ts_id <- "201708160208"
# model_name <- "Species - Fine Tune SS51 All"

# Camera Catalogue - Species Retrain SS 51 last layer
# project_id <- "camera_catalogue"
# model <- "cc_species_ss51_last_layer_only"
# project_name <- "Camera Catalogue"
# ts_id <- "201708151508"
# model_name <- "Species - Last Layer SS51 only"


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

# Elephant Expedition - Species Fine Tune SS 51 all
# project_id <- "elephant_expedition"
# model <- "ee_nonblank_no_ci_ss51_finetune_all"
# ts_id <- "201708150208"
# project_name <- "Elephant Expedition"
# model_name <- "Species - Fine Tune SS51 All"

# Elephant Expedition - Species Fine Tune SS 51 layst layer
# project_id <- "elephant_expedition"
# model <- "ee_nonblank_no_ci_ss51_last_layer_only"
# ts_id <- "201708142208"
# project_name <- "Elephant Expedition"
# model_name <- "Species - Last Layer SS51 only"

# Elephant Expedition - Species no dups
# project_id <- "elephant_expedition"
# model <- "ee_nonblank_no_cannotidentify_new_subject"
# ts_id <- "201708180508"
# project_name <- "Elephant Expedition"
# model_name <- "Species (excl. Cannotidentify & no duplicates)"

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
  path_main=path_main,
  n_classes=length(levels(preds$y_true))
)

output <- "pdf"

rmarkdown::render(input="analyses/report_template_debug.Rmd", 
                  params=params,
                  output_format = paste(output,"_document",sep=""),
                  output_file = paste(path_output_report,project_id,"_",model,".",output,sep=""))

