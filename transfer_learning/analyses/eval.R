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

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
project_id <- "ss"
#pred_file <- "ss_species_26_201707271307_preds_test"
pred_file = "ss_blank_vs_non_blank_small_201707271407_preds_test"
#log_file <- "ss_species_26_201707231807_training"
log_file <- "ss_blank_vs_non_blank_small_201707172207_training"
#model <- "ss_species_26"
model <- "blank_vs_non_blank_small"


# Elephant Expedition
path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
project_id <- "elephant_expedition"
pred_file = "ee_blank_vs_nonblank_201708012008_preds_test"
log_file <- "ee_blank_vs_nonblank_201708012008_training"
model <- "ee_blank_vs_nonblank"


############################ -
# Paths ----
############################ -

path_logs <- paste(path_main,"logs/",project_id,"/",sep="")
path_save <- paste(path_main,"save/",project_id,"/",sep="")



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
log_rf$set <- ifelse(grepl(pattern = "Train", log_rf$variable),"Test","Train")


gg <- ggplot(log_rf, aes(x=epoch, y=value, colour=set, group=variable)) + geom_line(lwd=1.5) +
  theme_light() +
  ggtitle(paste("Accuracy/Loss of Train / Test along training epochs\nmodel: ", model,sep="")) +
  xlab("Training Epoch") +
  ylab("Loss / Accuracy (%)") +
  facet_grid(group~., scales = "free") +
  scale_y_continuous(breaks=scales::pretty_breaks(n = 20)) +
  scale_color_brewer(type = "div", palette = "Set1")
gg

pdf(file = paste(path_save,model,"_log_file.pdf"), height=8, width=8)
gg
dev.off()
# win.metafile(file = paste(path_save,model,"_log_file.wmf"), width=4, height=4)
# gg
# dev.off()


############### -
# Plot Prediction Data
############### -

head(preds)

# total accuracy
sum(preds$y_true == preds$y_pred) / dim(preds)[1]



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
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip() +
  geom_text(size = 3, position = position_stack(vjust = 0.5))
gg

pdf(file = paste(path_save,model,"_classes_images.pdf"), height=8, width=7)
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
  geom_text(size = 3, position = position_stack(vjust = 0.5))
gg

pdf(file = paste(path_save,model,"_classes_subjects.pdf"), height=8, width=7)
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
  geom_text(size = 3, position = position_stack(vjust = 0.5))
gg

pdf(file = paste(path_save,model,"_classes_subjects_high_confidence.pdf"), height=8, width=7)
gg
dev.off()


# plot confusion matrix

# accuracy per true class
class_sum <- group_by(preds,y_true) %>% summarise(n_class = n())
conf <- group_by(preds,y_true, y_pred) %>% summarise(n = n()) %>% left_join(class_sum) %>%
  mutate(p_class=n / n_class)
conf
gg <- ggplot(conf, aes(x=y_pred, y=y_true)) + 
  geom_tile(aes(fill = p_class), colour = "black") + theme_minimal() +
  ggtitle(paste("Confusion Matrix\nmodel: ", model,sep="")) + 
  scale_fill_gradient2(low="red", mid="yellow", high="blue", midpoint=0.5) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_text(aes(label = round(p_class, 2)),cex=2)
gg


pdf(file = paste(path_save,model,"_confusion_matrix.pdf"), height=8, width=8)
gg
dev.off()


################################################ -
# Distribution of predicted values
################################################ -

preds_dist <- preds
preds_dist$correct <- ifelse(preds_dist$y_true==preds_dist$y_pred,1,0)
gg <- ggplot(preds_dist, aes(x=p, colour=correct)) + geom_histogram() + 
  facet_wrap("y_true") +
  ggtitle(paste("Prediction values distribution\nmodel: ", model,sep="")) + 
  theme_light()
gg

pdf(file = paste(path_save,model,"_dist_predictions.pdf"), height=8, width=8)
gg
dev.off()


