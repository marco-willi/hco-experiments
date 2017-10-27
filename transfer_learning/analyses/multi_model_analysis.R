############################ -
# Compare different models
# on the same dataset
############################ -

############################ -
# Libraries ----
############################ -

library(dplyr)
library(ggplot2)
library(stringr)
library(reshape2)

############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_output_report <- "D:/Studium_GD/Zooniverse/Results/transfer_learning/reports/"
path_output_report <- "D:\\Studium_GD\\Zooniverse\\Results\\transfer_learning\\reports\\"

############################ -
# Parameters ----
############################ -

model_comparisons <- list(
  data.frame(project_id="camera_catalogue",
             model="cc_species_v2",
             project_name="Camera Catalogue",
             ts_id = "201708210308",
             model_name="Species - 100%"), 
  data.frame(project_id="camera_catalogue",
             model="cc_species_ss51_last_layer_only",
             project_name="Camera Catalogue",
             ts_id = "201708212308",
             model_name="Species - 100% - TL from SS"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_75p_train_ss_last_layer",
             project_name="Camera Catalogue",
             ts_id = "201710161810",
             model_name="Species - 75% - TL from SS"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_50p_train_ss_last_layer",
             project_name="Camera Catalogue",
             ts_id = "201710170510",
             model_name="Species - 50% - TL from SS"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_25p_train_ss_last_layer",
             project_name="Camera Catalogue",
             ts_id = "201710171410",
             model_name="Species - 25% - TL from SS"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_12_5p_train_ss_last_layer",
             project_name="Camera Catalogue",
             ts_id = "201710180310",
             model_name="Species - 12.5% - TL from SS"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_75p_train",
             project_name="Camera Catalogue",
             ts_id = "201710150710",
             model_name="Species - 75%"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_50p_train",
             project_name="Camera Catalogue",
             ts_id = "201710152110",
             model_name="Species - 50%"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_25p_train",
             project_name="Camera Catalogue",
             ts_id = "201710160310",
             model_name="Species - 25%"),
  data.frame(project_id="camera_catalogue",
             model="cc_species_12_5p_train",
             project_name="Camera Catalogue",
             ts_id = "201710171810",
             model_name="Species - 12.5%")
)


############################ -
# Load all data ----
############################ -

define_paths <- function(project_id){
  path_logs <- paste(path_main,"logs/",project_id,"/",sep="")
  path_save <- paste(path_main,"save/",project_id,"/",sep="")
  path_db <- paste(path_main,"db/",project_id,"/",sep="")
  path_scratch <- paste(path_main,"scratch/",project_id,"/",sep="")
  path_figures <- paste(path_main,"save/",project_id,"/figures/",sep="")
  return(data.frame(path_logs, path_save, path_db, path_scratch, path_figures))
}


load_all_data <- function(model, ts_id, project_id){
  # define path
  paths <- define_paths(project_id)
  
  pred_file <- paste(model,"_",ts_id,"_preds_val",sep="")
  subject_set <- paste("val_subject_set_",model,sep="")
  log_file <- paste(model, "_",ts_id,"_training",sep="")
  
  preds <- read.csv(paste(paths$path_save,pred_file,".csv",sep=""))
  preds$model <- model
  
  # check levels
  missing_levels <- setdiff(levels(preds$y_true), levels(preds$y_pred))
  
  levels(preds$y_pred) <- c(levels(preds$y_pred),missing_levels)
  
  preds$correct <- factor(ifelse(preds$y_true==preds$y_pred,1,0))
  
  subjects <- jsonlite::read_json(paste(paths$path_db,subject_set,".json",sep=""), simplifyVector = FALSE, flatten=TRUE)
  head(subjects)
  
  log <- read.csv(paste(paths$path_logs,log_file,".log",sep=""))
  head(log)                  
  
  # remove top5 accuracy if less than 6 different classes
  if (nlevels(preds$y_true) < 6){
    log <- log[,!grepl("top_k",colnames(log))]
  }
  
  return(list(log=log, subjects=subjects, preds=preds))
  
}


all_data <- lapply(model_comparisons, function(x){
  l = load_all_data(x$model, x$ts_id, x$project_id)
  ll = c(l, x)})

all_data[[1]]$model_name
str_match(all_data[[1]]$model_name, "-(.*)%")
all_data[[1]]$model_name

############################ -
# Visualize ----
############################ -

# build log file stack
log_data_all <- lapply(all_data, function(x){
                          log = x$log
                          log$model = x$model
                          log$p_train = gsub(pattern = "- ", "", str_match(x$model_name,  "-(.*)%")[1])
                          log$train_mode = ifelse(grepl("ss", x$model),"Transfer-Learning", "Scratch")
                          return(log)})

log_data_all <- do.call(rbind, log_data_all)

log_data_all$p_train <- factor(log_data_all$p_train, levels = c("12.5%", "25%", "50%", "75%", "100%"))

# visualize p_train panels
ggplot(log_data_all, aes(x=epoch, y=val_acc, colour=train_mode)) + 
  geom_line(lwd=2) + theme_minimal() +
  facet_wrap("p_train") +
  xlab("Validation Accuracy") +
  ylab("Epoch") +
  scale_y_continuous(limits=c(min(log_data_all$val_acc),1))

# tranfer-learning / scratch comparison
ggplot(log_data_all, aes(x=epoch, y=val_acc, colour=p_train)) + 
  geom_line(lwd=2) + theme_minimal() +
  facet_wrap("train_mode") +
  ylab("Validation Accuracy") +
  xlab("Epoch") +
  scale_y_continuous(limits=c(min(log_data_all$val_acc),1)) +
  scale_color_brewer() +
  theme(text = element_text(size=20))


# by species
preds_data_all <- lapply(all_data, function(x){
  preds = x$preds
  preds$model = x$model[1]
  preds$p_train = gsub(pattern = "- ", "", str_match(x$model_name,  "-(.*)%")[1])[1]
  preds$train_mode = ifelse(grepl("ss", x$model),"Transfer-Learning", "Scratch")[1]
  class_dist <- dplyr::group_by(preds,model, p_train, train_mode, y_true) %>%
    summarise(n_obs = n(),matches = sum(y_true == y_pred), n = n()) %>%
    mutate(p_obs=n_obs / sum(n_obs), accuracy=matches/n)
  return(class_dist)})

preds_data_all  <- do.call(rbind, preds_data_all )
preds_data_all$p_train <- factor(preds_data_all$p_train, levels = c("12.5%", "25%", "50%", "75%", "100%"))


order_species <- group_by(preds_data_all, y_true) %>% summarise(n=max(n_obs)) %>% arrange(desc(n))

preds_data_all$y_true <- factor(preds_data_all$y_true, levels = order_species$y_true)

# number of observations
text_large <- 12
text_med <- 10
text_small <- 8
gg <- ggplot(data.frame(preds_data_all), aes(x=reorder(y_true, n_obs), y=n_obs, fill=p_train)) + geom_bar(stat="identity", position="dodge") +
  theme_light() +
  facet_wrap("train_mode") +
  #ggtitle(paste("Class Distribution \nmodel: ", model,sep="")) +
  ylab("# of observations") +
  xlab("") +
  coord_flip() +
  geom_text(aes(label=paste(" ",round(p_obs,4)*100," %",sep="")), size=text_med * (5/14), hjust="left") +
  scale_y_continuous(limits=c(0,max(preds_data_all$n_obs) * 1.1)) +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        strip.text.x = element_text(size = text_large, colour = "white", face="bold"))
gg



# accuracy per true class
gg <- ggplot(preds_data_all, aes(x=reorder(y_true, accuracy),y=accuracy, fill=p_train)) + 
  geom_bar(stat="identity", position="dodge") +
  theme_light() +
  facet_wrap("train_mode") +
  xlab("") +
  ylab("Accuracy") +
  coord_flip()  +
  # geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
  # geom_text(aes(label=paste(" Acc: ", sprintf("%.3f",round(accuracy,3)),"/ Obs: ", n), y=0.01), 
  #          size=text_small* (5/14), vjust="middle", hjust="left") +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        strip.text.x = element_text(size = text_large, colour = "white", face="bold")) +
  scale_y_continuous(limits=c(0,1)) +
  scale_fill_brewer()
gg



# difference in species
preds_data_all_diff <- dcast(data = preds_data_all, y_true + p_train ~ train_mode, value.var = "accuracy")
preds_data_all_diff$tl_minus_scratch <- preds_data_all_diff$`Transfer-Learning` - preds_data_all_diff$Scratch
preds_data_all_diff$y_true <- factor(preds_data_all_diff$y_true, levels=rev(order_species$y_true))
head(preds_data_all_diff)
gg <- ggplot(preds_data_all_diff, aes(x=y_true,y=tl_minus_scratch)) + 
  geom_bar(stat="identity", position="dodge") +
  theme_light() +
  facet_wrap("p_train") +
  xlab("") +
  ylab("Accuracy") +
  coord_flip()  +
  # geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
  # geom_text(aes(label=paste(" Acc: ", sprintf("%.3f",round(accuracy,3)),"/ Obs: ", n), y=0.01), 
  #          size=text_small* (5/14), vjust="middle", hjust="left") +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        strip.text.x = element_text(size = text_large, colour = "white", face="bold")) +
  scale_y_continuous(limits=c(-1,1)) +
  scale_fill_brewer()
gg




preds_data_all_diff <- dcast(data = preds_data_all, y_true + p_train ~ train_mode, value.var = "accuracy")
preds_data_all_diff$tl_minus_scratch <- preds_data_all_diff$`Transfer-Learning` - preds_data_all_diff$Scratch
preds_data_all_diff$y_true <- factor(preds_data_all_diff$y_true, levels=order_species$y_true)
head(preds_data_all_diff)
gg <- ggplot(preds_data_all_diff, aes(x=p_train,y=tl_minus_scratch)) + 
  geom_bar(stat="identity", position="dodge") +
  theme_light() +
  facet_wrap("y_true") +
  xlab("") +
  ylab("Accuracy") +
  coord_flip()  +
  # geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
  # geom_text(aes(label=paste(" Acc: ", sprintf("%.3f",round(accuracy,3)),"/ Obs: ", n), y=0.01), 
  #          size=text_small* (5/14), vjust="middle", hjust="left") +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        strip.text.x = element_text(size = text_large, colour = "white", face="bold")) +
  scale_y_continuous(limits=c(-1,1)) +
  scale_fill_brewer()
gg







