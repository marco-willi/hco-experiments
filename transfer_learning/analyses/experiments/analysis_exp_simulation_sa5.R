############################ -
# Libraries ----
############################ -

library(dplyr)
library(ggplot2)
library(reshape2)
library(jpeg)
library(grid)
library(gridExtra)

############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_project <- "db/camcat2/"
fname <- "classifications_experiment_20171012_exp_simulation.csv"
path_scratch <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/scratch/camcat2/"
path_save <- "D:/Studium_GD/Zooniverse/Project/Poster/CS_PosterFair_Fall2017/"

############################ -
# Read Classifications ----
############################ -

data_raw <- read.csv2(paste(path_main, path_project, fname, sep=""),sep = ",", quote = "\"", header = TRUE)
head(data_raw)
names(data_raw)

############################ -
# Categorize data ----
############################ -

matcher <- function(y_exp, y_plur){
  match <- sapply(1:length(y_exp), function(x){
    grepl(pattern = y_exp[x], x = y_plur[x])
  })
  return(match)
}


data <- select(data_raw, X, subject_id, machine_label, machine_prob, retire_label_exp,
               label_plur, retirement_reason_plur, X.experiment_group, link) %>%
  filter(retire_label_exp != "" & retirement_reason_plur != 'Not Retired') %>% 
  mutate(x_match_exp_plur = matcher(retire_label_exp, label_plur),
         x_match_mach_plur = matcher(machine_label, label_plur),
         x_different_outcome = ifelse(retire_label_exp != 'no_agreement' & !x_match_exp_plur,1,0))
  
table(data$x_match_exp_plur, data$X.experiment_group, data$x_match_mach_plur)

filter(data, x_different_outcome==1)


ggplot(data, aes(x=x_match_exp_plur, fill=factor(X.experiment_group))) + facet_grid(x_different_outcome~x_match_mach_plur, scales="free") + geom_bar(position="dodge")






