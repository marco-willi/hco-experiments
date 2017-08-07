############################ -
# Libraries ----
############################ -

library(ggplot2)
library(reshape2)
library(plyr)
library(dplyr)


############################ -
# Parameters ----
############################ -


# Snapshot Serengeti - Top26 species
path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
project_id <- "ss"
pred_file <- "ss_species_26_201707271307_preds_test"
log_file <- "ss_species_26_201707231807_training"
model <- "ss_species_26"



# Snapshot Serengeti - Blank vs Non Blank
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "ss"
# pred_file = "ss_blank_vs_non_blank_small_201707271407_preds_test"
# log_file <- "ss_blank_vs_non_blank_small_201707172207_training"
# model <- "blank_vs_non_blank_small"


# Elephant Expedition - blank vs non-blank
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_blank_vs_nonblank_201708021608_preds_test"
# log_file <- "ee_blank_vs_nonblank_201708012008_training"
# model <- "ee_blank_vs_nonblank"

# Elephant Expedition - species
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_nonblank_201708030208_preds_test"
# log_file <- "ee_nonblank_201708021908_training"
# model <- "ee_nonblank"

# Elephant Expedition - species no cannotidentify
# path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
# project_id <- "elephant_expedition"
# pred_file = "ee_nonblank_no_cannotidentify_201708050608_preds_val"
# log_file <- "ee_nonblank_no_cannotidentify_201708042308_training"
# model <- "ee_nonblank_no_cannotidentify"


############################ -
# Paths ----
############################ -

path_logs <- paste(path_main,"logs/",project_id,"/",sep="")
path_save <- paste(path_main,"save/",project_id,"/",sep="")
path_figures <- paste(path_main,"save/",project_id,"/figures/",sep="")



############################ -
# Read Data ----
############################ -

preds <- read.csv(paste(path_save,pred_file,".csv",sep=""))
preds$model <- model
head(preds)

log <- read.csv(paste(path_logs,log_file,".log",sep=""))
head(log)                  

# remove top5 accuracy if less than 6 different classes
if (nlevels(preds$y_true) < 6){
  log <- log[,!grepl("top_k",colnames(log))]
}

############################ -
# Plot Data ----
############################ -

############### -
# Plot LOG File
############### -

# reformat data
log_rf <- melt(log, id.vars = c("epoch"))
log_rf$group <- ifelse(grepl(pattern = "loss", log_rf$variable),"Loss",
                       ifelse(grepl(pattern = "top", log_rf$variable),"Top-5 Accuracy","Top-1 Accuracy"))

#log_rf <- filter(log_rf, !grepl(pattern = "loss", variable))
head(log_rf)

# rename variables
str(log_rf)
log_rf$variable <- revalue(log_rf$variable, c("acc"="Top-1 Accuracy - Train", "loss"="Train Loss",
                          "val_acc"="Top-1 Accuracy - Test", "val_loss"="Test Loss",
                          "sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Train",
                          "val_sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Test"))
log_rf$set <- ifelse(grepl(pattern = "Train", log_rf$variable),"Train","Test")


gg <- ggplot(log_rf, aes(x=epoch, y=value, colour=set, group=variable)) + geom_line(lwd=1.5) +
  theme_light() +
  ggtitle(paste("Accuracy/Loss of Train / Test along training epochs\nmodel: ", model,sep="")) +
  xlab("Training Epoch") +
  ylab("Loss / Accuracy (%)") +
  facet_grid(group~., scales = "free") +
  scale_y_continuous(breaks=scales::pretty_breaks(n = 10)) +
  scale_color_brewer(type = "div", palette = "Set1")
gg


print_name = paste(path_save,model,"_log_file",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=12,units = "cm", res=128)
gg
dev.off()

############### -
# Plot Prediction Data
############### -

head(preds)

# check levels
missing_levels <- setdiff(levels(preds$y_true), levels(preds$y_pred))

levels(preds$y_pred) <- c(levels(preds$y_pred),missing_levels)

# total accuracy
sum(preds$y_true == preds$y_pred) / dim(preds)[1]

#########################
# Class distribution
#########################

class_dist <- group_by(preds,y_true) %>%
  summarise(n_obs = n()) %>%
  mutate(p_obs=n_obs / sum(n_obs))

gg <- ggplot(class_dist, aes(x=reorder(y_true, n_obs), y=n_obs)) + geom_bar(stat="identity") +
  theme_light() +
  ggtitle(paste("Class Distribution \nmodel: ", model,sep="")) +
  ylab("# of samples") +
  xlab("") +
  coord_flip() +
  geom_text(aes(label=paste(" ",round(p_obs,4)*100," %",sep="")), hjust="left") +
  scale_y_continuous(limits=c(0,max(class_dist$n_obs) * 1.1))
gg


print_name = paste(path_save,model,"_classes_numbers",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=7)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=10,units = "cm", res=128)
gg
dev.off()

################################################ -
# Per Class & Image Accuracy
################################################ -

# accuracy per true class
preds_class <- group_by(preds,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, label=paste("Acc: ", round(accuracy,3)," Obs: ", n))) + 
  geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Accuracy for Classes\nmodel: ", model,"\nall images",sep="")) +
  xlab("") +
  ylab("Accuracy (%)") +
  coord_flip() +
  geom_text(size = 3, position = position_stack(vjust = 0.5), colour="white")
gg


print_name = paste(path_figures,model,"_classes_images",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=7)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=10,units = "cm", res=128)
gg
dev.off()

################################################ -
# Per Class & Most confidence Image Accuracy
################################################ -

# take only with most confidence
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))
preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p)
head(preds_1)

# accuracy per true class
preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, label=paste("Acc: ", round(accuracy,3)," Obs: ", n))) + 
  geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Accuracy for Classes\nmodel: ", model,"\nSubject level",sep="")) +
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip() +
  geom_text(size = 3, position = position_stack(vjust = 0.5), colour="white")
gg


print_name = paste(path_figures,model,"_classes_subjects",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=7)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=10,units = "cm", res=128)
gg
dev.off()

################################################ -
# Per Class & Most confidence Image Accuracy &
# only if above 95% model score
################################################ -

# take only with most confidence and Threshold
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))
preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>0.95)
head(preds_1)

# accuracy per true class
preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

# get total class numbers and join
class_numbers <- group_by(preds, y_true) %>% summarise(n_total=n_distinct(subject_id))

preds_class <- left_join(preds_class, class_numbers, by="y_true") %>% mutate(p_high_threshold=round(n/n_total,2)*100)

gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, 
                              label=paste("Acc: ", round(accuracy,3)," Obs: ", n," / ",n_total," (",p_high_threshold," %)",sep=""))) + 
  geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Accuracy for Classes\nmodel: ", model,"\nsubject level and only > 95% confidence",sep="")) +
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip() +
  geom_text(size = 3, position = position_stack(vjust = 0.5), colour="white")
gg


print_name = paste(path_figures,model,"_classes_subjects_high_confidence",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=7)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=10,units = "cm", res=128)
gg
dev.off()


################################################ -
# Convidence vs Completeness vs Accuracy ----
################################################ -


# Overall View

# take only with most confidence and Threshold
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))

res <- NULL

# thresholds to test
thresholds <- seq(min(preds$p),0.99,by=0.025)
for (ii in seq_along(thresholds)){
  
  preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>=thresholds[ii])
  head(preds_1)
  
  # accuracy overall
  preds_class <- group_by(preds_1) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
    mutate(accuracy=matches/n)
  preds_class
  
  # get total class numbers and join
  class_numbers <- group_by(preds) %>% summarise(n_total=n_distinct(subject_id))
  
  preds_class$p_high_threshold <- round(preds_class$n/class_numbers$n_total,2)
  preds_class$threshold <- thresholds[ii]
  
  res[[ii]] <- preds_class
}
res2 <- do.call(rbind,res)
head(res2)
res3 <- melt(data = res2, id.vars = c("threshold")) %>% filter(variable %in% c("accuracy", "p_high_threshold"))
head(res3)

gg <- ggplot(res3, aes(x=threshold, y=value, colour=variable, group=variable)) + geom_line(lwd=2)  + 
  theme_light() +
  ggtitle(paste("Accuracy vs Modle Threshold\nmodel: ", model,sep="")) + 
  xlab("Model Threshold") +
  ylab("Accuracy / Share (%)") +
  scale_x_continuous(limit = c(min(res3$threshold),1)) +
  scale_y_continuous(breaks = seq(min(res3$value),1,0.02)) +
  scale_color_brewer(type = "qual", guide =  guide_legend(title=NULL),  
                     labels = c("Accuracy (%)", "Proportion of Images\nAbove Threshold (%)")) +
  theme(axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12))


print_name = paste(path_figures,model,"_accuracy_vs_threshold_overall",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=6, width=12)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=16, height=10,units = "cm", res=128)
gg
dev.off()



# take only with most confidence and Threshold
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))

res <- NULL

# thresholds to test
thresholds <- seq(min(preds$p),0.95,by=0.05)
for (ii in seq_along(thresholds)){

  preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>=thresholds[ii])
  head(preds_1)
  
  # accuracy per true class
  preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
    mutate(accuracy=matches/n)
  preds_class
  
  # get total class numbers and join
  class_numbers <- group_by(preds, y_true) %>% summarise(n_total=n_distinct(subject_id))
  
  preds_class <- left_join(preds_class, class_numbers, by="y_true") %>% mutate(p_high_threshold=round(n/n_total,2))
  preds_class$threshold <- thresholds[ii]
  preds_class$p_high_threshold <- ifelse(preds_class$p_high_threshold>1,1,preds_class$p_high_threshold)
  
  res[[ii]] <- preds_class
}
res2 <- do.call(rbind,res)
res3 <- melt(data = res2, id.vars = c("threshold", "y_true")) %>% filter(variable %in% c("accuracy", "p_high_threshold"))

gg <- ggplot(res3, aes(x=threshold, y=value, colour=variable, group=variable)) + geom_line(lwd=2)  + 
  theme_light() +
  facet_wrap("y_true") +
  ggtitle(paste("Accuracy vs Modle Threshold\nmodel: ", model,sep="")) + 
  xlab("Model Threshold") +
  ylab("Accuracy / Share (%)") +
  scale_x_continuous(limit = c(min(res3$threshold),1)) +
  scale_y_continuous(breaks = seq(min(res3$value),1,0.1)) +
  scale_color_brewer(type = "qual", guide =  guide_legend(title=NULL),  
                     labels = c("Accuracy (%)", "Proportion of Images\nAbove Threshold (%)")) +
  theme(axis.text = element_text(size=12),
        axis.title = element_text(size=14),
        legend.text = element_text(size=12))
gg

print_name = paste(path_figures,model,"_accuracy_vs_threshold_per_class",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=14, width=14)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=32, height=28,units = "cm", res=128)
gg
dev.off()
################################################ -
# Confusion Matrix ----
################################################ -

# empty confusion matrix
conf_empty <- expand.grid(levels(preds$y_true),levels(preds$y_true))
names(conf_empty) <- c("y_true","y_pred")

# accuracy per true class
class_sum <- group_by(preds,y_true) %>% summarise(n_class = n())
conf <- group_by(preds,y_true, y_pred) %>% summarise(n = n()) %>% left_join(class_sum) %>%
  mutate(p_class=n / n_class)
conf <- left_join(conf_empty, conf,by=c("y_true","y_pred")) %>% mutate(p_class = ifelse(is.na(p_class),0,p_class))

conf
gg <- ggplot(conf, aes(x=y_pred, y=y_true)) + 
  geom_tile(aes(fill = p_class), colour = "black") + theme_bw() +
  ggtitle(paste("Confusion Matrix\nmodel: ", model,sep="")) + 
  scale_fill_gradient2(low="blue", mid="yellow", high="red", midpoint=0.5, guide =  FALSE) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_text(aes(label = round(p_class, 2)),cex=2.5) +
  ylab("True") +
  xlab("Predicted") +
  scale_x_discrete()
gg


print_name = paste(path_figures,model,"_confusion_matrix",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=12,units = "cm", res=128)
gg
dev.off()

################################################ -
# Distribution of predicted values
################################################ -

preds_dist <- preds
preds_dist$correct <- factor(ifelse(preds_dist$y_true!=preds_dist$y_pred,0,1))
gg <- ggplot(preds_dist, aes(x=p, fill=correct)) + geom_density(alpha=0.3) + 
  facet_wrap("y_true", scales="free") +
  ggtitle(paste("Predicted values - density distribution\nmodel: ", model,sep="")) + 
  theme_light() +
  xlab("Predicted Value") +
  ylab("Density") + 
  scale_fill_brewer(type = "qual",direction = -1)
gg

print_name = paste(path_figures,model,"_dist_predictions",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=12,units = "cm", res=128)
gg
dev.off()

