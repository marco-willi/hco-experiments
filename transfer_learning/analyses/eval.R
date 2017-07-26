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
pred_file <- "ss_species_26_201707261907_preds_test"
log_file <- "ss_species_26_201707231807_training"
model <- "ss_species_26"

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

# Plot LOG File

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
  scale_color_brewer(palette="Set1")
gg


# Plot Prediction Data
head(preds)

# accuracy per true class
preds_class <- group_by(preds,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

gg <- ggplot(preds_class, aes(x=y_true,y=accuracy)) + geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Accuracy for Classes\nmodel: ", model,sep="")) +
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip()
gg

# take only with most confidence
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))
preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p)
head(preds_1)

# accuracy per true class
preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

gg <- ggplot(preds_class, aes(x=y_true,y=accuracy)) + geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Top Accuracy for Classes - Aggregated\nmodel: ", model,sep="")) +
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip()
gg


# take only with most confidence and Threshold
preds_max_p <- group_by(preds, subject_id) %>% summarise(max_p=max(p))
preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>0.95)
head(preds_1)

# accuracy per true class
preds_class <- group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
  mutate(accuracy=matches/n)
preds_class

gg <- ggplot(preds_class, aes(x=y_true,y=accuracy)) + geom_bar(stat="identity", colour="gray") +
  theme_light() +
  ggtitle(paste("Test Top Accuracy for Classes - only > 95% confidence\nmodel: ", model,sep="")) +
  xlab("Species") +
  ylab("Accuracy (%)") +
  coord_flip()
gg



