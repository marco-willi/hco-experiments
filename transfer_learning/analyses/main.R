############################ -
# Parameters ----
############################ -


# Snapshot Serengeti - Top26 species
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "ss"
# pred_file <- "ss_species_26_201707271307_preds_test"
# log_file <- "ss_species_26_201707231807_training"
# model <- "ss_species_26"



# Snapshot Serengeti - Blank vs Non Blank
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "ss"
# pred_file = "ss_blank_vs_non_blank_small_201707271407_preds_test"
# log_file <- "ss_blank_vs_non_blank_small_201707172207_training"
# model <- "blank_vs_non_blank_small"


# Elephant Expedition - blank vs non-blank
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_blank_vs_nonblank_201708021608_preds_val"
# log_file <- "ee_blank_vs_nonblank_201708012008_training"
# model <- "ee_blank_vs_nonblank"
# subject_set <- "val_subject_set_ee_blank_vs_nonblank"

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
path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
project_id <- "camera_catalogue"
pred_file <- "cc_blank_vehicle_species_201708052008_preds_val"
log_file <- "cc_blank_vehicle_species_201708052008_training"
model <- "cc_blank_vehicle_species"
subject_set <- "val_subject_set_cc_blank_vehicle_species"


# Load Functions
source("analyses/plot_functions.R")

# Load data
source("analyses/load_data.R")

# save eval plots
source("analyses/eval2.R")

# save subject set plots
source("analyses/plot_subject_set.R")