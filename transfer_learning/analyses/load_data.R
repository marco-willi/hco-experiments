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


log <- read.csv(paste(path_logs,log_file,".log",sep=""))
head(log)                  

# remove top5 accuracy if less than 6 different classes
if (nlevels(preds$y_true) < 6){
  log <- log[,!grepl("top_k",colnames(log))]
}


