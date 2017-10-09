############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_project <- "db/camera_catalogue/"
file_in = "CHECK2_manifest_set_28_2017.09.12.csv"

raw_file <- read.csv(paste(path_main, path_project, file_in, sep=""))
names(raw_file) <- sapply(names(raw_file), function(x) { gsub(pattern = "X.", replacement = "",x)})
names(raw_file)

table(raw_file$experiment_group)

############################ -
# Analyse ----
############################ -

library(ggplot2)
library(dplyr)

gg <- ggplot(filter(raw_file,experiment_group=="1") , aes(x=machine_probability)) + geom_density() + facet_wrap("machine_prediction", scales="free")
gg

class_dist <- filter(raw_file,experiment_group=="1") %>% group_by(machine_prediction) %>% summarise(n_preds=n()) %>%
  mutate(p_preds=n_preds / sum(n_preds))
gg <- ggplot(class_dist , aes(x=reorder(machine_prediction, n_preds), y=n_preds)) + geom_bar(stat="identity") +  theme_light() +
  ggtitle("Predicted Classes") +
  ylab("# predicted") +
  xlab("") +
  coord_flip() +
  scale_y_continuous(limits=c(0,max(class_dist$n_preds) * 1.1)) +
  geom_text(aes(label=paste(" ",round(p_preds,4)*100," %",sep="")), hjust="left") 
gg


plot_class_dist <- function(preds, model){
  class_dist <- dplyr::group_by(preds,y_true) %>%
    summarise(n_obs = n()) %>%
    mutate(p_obs=n_obs / sum(n_obs))
  
  gg <- ggplot(class_dist, aes(x=reorder(y_true, n_obs), y=n_obs)) + geom_bar(stat="identity", fill="wheat") +
    theme_light() +
    ggtitle(paste("Class Distribution \nmodel: ", model,sep="")) +
    ylab("# of observations") +
    xlab("") +
    coord_flip() +
    geom_text(aes(label=paste(" ",round(p_obs,4)*100," %",sep="")), size=text_med * (5/14), hjust="left") +
    scale_y_continuous(limits=c(0,max(class_dist$n_obs) * 1.1)) +
    theme(axis.text = element_text(size=text_large),
          axis.title = element_text(size=text_large),
          strip.text.x = element_text(size = text_large, colour = "white", face="bold"))
  return(gg)
  