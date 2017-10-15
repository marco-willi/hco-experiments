

############################ -
# Read Data ----
############################ -

pred_file <- paste(model,"_",ts_id,"_preds_val",sep="")
subject_set <- paste("val_subject_set_",model,sep="")
log_file <- paste(model, "_",ts_id,"_training",sep="")

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

# 
# # replace levels
# replace_level <- function(lvls, old, new){
#   if (old %in% lvls){
#     lvls[which(lvls == old)] <- new
#   }
#   return(lvls)
# }
# 
# lvls <- levels(preds$y_true)
# lvls <- replace_level(lvls, "SQUIRRELSANDCHIPMUNKS", "SQUIRREL")
# lvls <- replace_level(lvls, "OTHERSMALLMAMMAL", "OTHSMALL")
# levels(preds$y_true) <- lvls
# 
# lvls <- levels(preds$y_pred)
# lvls <- replace_level(lvls, "SQUIRRELSANDCHIPMUNKS", "SQUIRREL")
# lvls <- replace_level(lvls, "OTHERSMALLMAMMAL", "OTHSMALL")
# levels(preds$y_pred) <- lvls
