
############################ -
# Plot Data ----
############################ -

############### -
# Plot LOG File
############### -

gg <- plot_log(log, model=model)

print_name = paste(path_figures,model,"_log_file",sep="")
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

gg <- plot_class_dist(preds, model)

print_name = paste(path_figures,model,"_classes_numbers",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=7)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=10,units = "cm", res=128)
gg
dev.off()

################################################ -
# Per Class & Image Accuracy
################################################ -

gg <- plot_class_acc(preds, model)
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
gg <- plot_class_most_conf_acc(preds, model)

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

gg <- plot_class_most_conf_95th_acc(preds, model)
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

gg <- plot_threshold_vs_acc_overall(preds, model)
gg

print_name = paste(path_figures,model,"_accuracy_vs_threshold_overall",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=6, width=12)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=16, height=10,units = "cm", res=128)
gg
dev.off()


gg <- plot_threshold_vs_acc_class(preds, model)

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
gg <- plot_cm(preds, model)

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

gg <- plot_dist_pred(preds, model)

print_name = paste(path_figures,model,"_dist_predictions",sep="")
pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
gg
dev.off()
png(file = paste(print_name,".png",sep=""), width=12, height=12,units = "cm", res=128)
gg
dev.off()

