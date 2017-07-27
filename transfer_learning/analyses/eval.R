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


############################ -
# Plot Data ----
############################ -

############### -
# Plot LOG File
############### -

# reformat data
log_rf <- melt(log, id.vars = c("epoch"))
log_rf <- filter(log_rf, !grepl(pattern = "loss", variable))
head(log_rf)

# rename variables
str(log_rf)
log_rf$variable <- revalue(log_rf$variable, c("acc"="Top-1 Accuracy - Train", "loss"="Train Loss",
                          "val_acc"="Top-1 Accuracy - Test", "val_loss"="Test Loss",
                          "sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Train",
                          "val_sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Test"))


gg <- ggplot(log_rf, aes(x=epoch, y=value, colour=variable)) + geom_line(lwd=1.5) +
  theme_light() +
  ggtitle(paste("Accuracy Train / Test along training epochs\nmodel: ", model,sep="")) +
  xlab("Epoch") +
  ylab("Accuracy (%)") +
  scale_color_brewer("",palette="Set1")
gg

pdf(file = paste(path_save,model,"_log_file.pdf"), height=5, width=8)
gg
dev.off()


############### -
# Plot Prediction Data
############### -

head(preds)

# total accuracy
sum(preds$y_true == preds$y_pred) / dim(preds)[1]

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

# take only with most confidence and Threshold
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))
preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>0.95)
head(preds_1)

# accuracy per true class
preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class


gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, label=paste("Acc: ", round(accuracy,3)," Obs: ", n))) + 
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


