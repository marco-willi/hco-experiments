############################ -
# Libraries ----
############################ -

library(ggplot2)
library(reshape2)
library(plyr)
library(dplyr)
library(jsonlite)
library(jpeg)
library(grid)
library(gridExtra)
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
path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
project_id <- "elephant_expedition"
pred_file = "ee_blank_vs_nonblank_201708021608_preds_val"
log_file <- "ee_blank_vs_nonblank_201708012008_training"
model <- "ee_blank_vs_nonblank"
subject_set <- "val_subject_set_ee_blank_vs_nonblank"

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

############################ -
# Paths ----
############################ -

path_logs <- paste(path_main,"logs/",project_id,"/",sep="")
path_save <- paste(path_main,"save/",project_id,"/",sep="")
path_db <- paste(path_main,"db/",project_id,"/",sep="")
path_scratch <- paste(path_main,"scratch/",project_id,"/",sep="")
path_figures <- paste(path_main,"save/",project_id,"/figures/",sep="")

############################ -
# Read Data ----
############################ -
preds <- read.csv(paste(path_save,pred_file,".csv",sep=""))
preds$model <- model

# check levels
missing_levels <- setdiff(levels(preds$y_true), levels(preds$y_pred))

levels(preds$y_pred) <- c(levels(preds$y_pred),missing_levels)

preds$correct <- factor(ifelse(preds$y_true==preds$y_pred,1,0))

subjects <- jsonlite::read_json(paste(path_db,subject_set,".json",sep=""), simplifyVector = FALSE, flatten=TRUE)
head(subjects)

############################ -
# Find some missclassifications ----
############################ -

random_wrongs <- filter(preds,correct==0)
set.seed(23)
random_wrongs <- random_wrongs[sample(size = 10,x=dim(random_wrongs)[1]),]
random_wrongs


############################ -
# Plot missclassifications ----
############################ -

for (ii in 1:10){
  
  id <- paste(random_wrongs[ii,"subject_id"])
  preds <- random_wrongs[ii,"preds_all"]
  preds <- fromJSON(paste("[",gsub(pattern = "'", "\"", x=as.character(preds[[1]])),"]",sep=""))
  preds <- melt(preds,value.name = "prob",variable.name = "class")
  sub <- subjects[id]
  url <- unlist(sub[[id]]['urls'])
  label <- unlist(sub[[id]]['label'])
  url
  label
  preds
  file_name <- paste(path_scratch,"image_",ii,".jpeg",sep="")
  download.file(url, destfile = file_name, mode = 'wb')
  
  
  img <- readJPEG(file_name)
  
  
  gg1 <- ggplot(data.frame(x=0:1,y= 0:1),aes(x=x,y=y), geom="blank") +
    annotation_custom(rasterGrob(img, width=unit(1,"npc"), height=unit(1,"npc")), 
                      -Inf, Inf, -Inf, Inf) + theme_minimal() +
    theme(axis.title = element_blank(), axis.text = element_blank()) +
    theme(plot.margin = unit(c(0.7,0.7,0,0.7), "cm"))
  
  gg2 <- ggplot(preds, aes(x=reorder(class, prob),y=prob)) + geom_bar(stat="identity", fill="lightblue") +
    coord_flip() +
    theme_light() +
    ylab("Predicted Probability") +
    xlab("") +
    theme(axis.text.y=element_blank(), axis.text.x=element_text(size=16),
          axis.title.x=element_text(size=16),
          axis.title.y=element_text(size=16),
          axis.ticks.y = element_blank()) +
    geom_text(aes(label=class, y=0.05), size=5,fontface="bold", vjust="middle", hjust="left") +
    theme(plot.margin = unit(c(0.7,0.7,0,0.7), "cm")) +
    scale_y_continuous(expand=c(0,0), limits = c(0,1)) +
    labs(x=NULL)

  
  title=textGrob(label = label,gp=gpar(fontsize=20,fontface="bold"), vjust=1)
  
  print_name = paste(path_figures,model,"_sample_wrong_",ii,sep="")
  pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
  grid.arrange(gg1,gg2,top=title)
  dev.off()
  png(file = paste(print_name,".png",sep=""), width=18, height=18,units = "cm", res=128)
  grid.arrange(gg1,gg2,top=title)
  dev.off()
}

