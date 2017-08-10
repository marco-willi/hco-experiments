############################ -
# Libraries ----
############################ -

library(ggplot2)
library(reshape2)
library(jsonlite)
library(jpeg)
library(grid)
library(gridExtra)
library(plyr)
library(dplyr)

########################### -
# Plot Log File ----
########################### -

plot_log <- function(log, model=""){
  # reformat data
  log_rf <- melt(log, id.vars = c("epoch"))
  log_rf$group <- ifelse(grepl(pattern = "loss", log_rf$variable),"Loss",
                         ifelse(grepl(pattern = "top", log_rf$variable),"Top-5 Accuracy","Top-1 Accuracy"))
  log_rf$variable <- revalue(log_rf$variable, c("acc"="Top-1 Accuracy - Train", "loss"="Train Loss",
                                                "val_acc"="Top-1 Accuracy - Validation", "val_loss"="Validation Loss",
                                                "sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Train",
                                                "val_sparse_top_k_categorical_accuracy"="Top-5 Accuracy - Validation"))
  log_rf$set <- ifelse(grepl(pattern = "Train", log_rf$variable),"Train","Validation")
  
  gg <- ggplot(log_rf, aes(x=epoch, y=value, colour=set, group=variable)) + geom_line(lwd=1.5) +
    theme_light() +
    ggtitle(paste("Accuracy/Loss of Train / Validation along training epochs\nmodel: ", model,sep="")) +
    xlab("Training Epoch") +
    ylab("Accuracy (%) / Loss") +
    facet_grid(group~., scales = "free") +
    scale_y_continuous(breaks=scales::pretty_breaks(n = 10)) +
    scale_x_continuous(breaks=scales::pretty_breaks(n=10)) +
    scale_color_brewer(type = "div", palette = "Set1") +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          strip.text.y = element_text(size = 14, colour = "white", face="bold"),
          legend.text = element_text(size=12),
          legend.position = "bottom",
          legend.box = "horizontal",
          legend.title = element_blank(),
          legend.background = element_rect(size=1,colour="black"))
  return(gg)
}

########################### -
# Plot Class Distr ----
########################### -

plot_class_dist <- function(preds, model){
  class_dist <- dplyr::group_by(preds,y_true) %>%
    summarise(n_obs = n()) %>%
    mutate(p_obs=n_obs / sum(n_obs))
  
  gg <- ggplot(class_dist, aes(x=reorder(y_true, n_obs), y=n_obs)) + geom_bar(stat="identity") +
    theme_light() +
    ggtitle(paste("Class Distribution \nmodel: ", model,sep="")) +
    ylab("# of samples") +
    xlab("") +
    coord_flip() +
    geom_text(aes(label=paste(" ",round(p_obs,4)*100," %",sep="")), hjust="left") +
    scale_y_continuous(limits=c(0,max(class_dist$n_obs) * 1.1)) +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}


################################################ -
# Plot Class & Image Accuracy ----
################################################ -

plot_class_acc <- function(preds, model){
  # accuracy per true class
  preds_class <- dplyr::group_by(preds,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
    mutate(accuracy=matches/n)
  preds_class
  
  gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, label=paste("Acc: ", round(accuracy,3)," Obs: ", n))) + 
    geom_bar(stat="identity", colour="gray") +
    theme_light() +
    ggtitle(paste("Validation Accuracy for Classes\nmodel: ", model,"\nall images",sep="")) +
    xlab("") +
    ylab("Accuracy (%)") +
    coord_flip()  +
    geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}



################################################ -
# Plot Class & Most confidence Image Accuracy
################################################ -

plot_class_most_conf_acc <- function(preds, model){
  # take only with most confidence
  preds_max_p <- dplyr::group_by(preds, subject_id) %>% summarise(max_p=max(p))
  preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p)
  
  # accuracy per true class
  preds_class <- dplyr::group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
    mutate(accuracy=matches/n)
  
  gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, label=paste("Acc: ", round(accuracy,3)," Obs: ", n))) + 
    geom_bar(stat="identity", colour="gray") +
    theme_light() +
    ggtitle(paste("Validation Accuracy for Classes\nmodel: ", model,"\nSubject level",sep="")) +
    xlab("Species") +
    ylab("Accuracy (%)") +
    coord_flip()  +
    geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}



################################################ -
# Per Class & Most confidence Image Accuracy &
# only if above 95% model score
################################################ -

plot_class_most_conf_95th_acc<- function(preds, model){
  # take only with most confidence and Threshold
  preds_max_p <- dplyr::group_by(preds, subject_id) %>% summarise(max_p=max(p))
  preds_1 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% filter(p>0.95)
  
  # accuracy per true class
  preds_class <- dplyr::group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
    mutate(accuracy=matches/n)
  
  # get total class numbers and join
  class_numbers <- dplyr::group_by(preds, y_true) %>% summarise(n_total=n_distinct(subject_id))
  
  preds_class <- left_join(preds_class, class_numbers, by="y_true") %>% mutate(p_high_threshold=round(n/n_total,2)*100)
  
  gg <- ggplot(preds_class, aes(x=reorder(y_true, accuracy),y=accuracy, 
                                label=paste("Acc: ", round(accuracy,3)," Obs: ", n," / ",n_total," (",p_high_threshold," %)",sep=""))) + 
    geom_bar(stat="identity", colour="gray") +
    theme_light() +
    ggtitle(paste("Validation Accuracy for Classes\nmodel: ", model,"\nsubject level and only > 95% confidence",sep="")) +
    xlab("Species") +
    ylab("Accuracy (%)") +
    coord_flip() +
    geom_text(size = 4, position = position_stack(vjust = 0.5), colour="white") +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}



################################################ -
# Convidence vs Completeness vs Accuracy ----
################################################ -

plot_threshold_vs_acc_overall<- function(preds, model){
  
  # take only with most confidence and Threshold
  preds_max_p <- dplyr::group_by(preds, subject_id) %>% summarise(max_p=max(p)) 
  preds_0 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% dplyr::arrange(subject_id)
  # only take one image
  preds_min_image_id <- group_by(preds_0, subject_id) %>% summarise(image_id=min(image_id))
  preds_0 <- inner_join(preds_0,preds_min_image_id,by=c("subject_id","image_id"))
  res <- NULL
  
  # thresholds to test
  thresholds <- seq(min(preds$p),0.99,by=0.025)
  for (ii in seq_along(thresholds)){
    
    preds_1 <- preds_0 %>% filter(p>=thresholds[ii])
    
    # accuracy overall
    preds_class <- dplyr::group_by(preds_1) %>% summarise(matches = sum(y_true == y_pred), n = n_distinct(subject_id)) %>%
      mutate(accuracy=matches/n)
    
    # get total class numbers and join
    class_numbers <- dplyr::group_by(preds) %>% summarise(n_total=n_distinct(subject_id))
    
    preds_class$p_high_threshold <- sapply(round(preds_class$n/class_numbers$n_total,4),function(x){min(1,x)})
    preds_class$threshold <- thresholds[ii]
    
    res[[ii]] <- preds_class
  }
  res2 <- do.call(rbind,res)
  res3 <- melt(data = res2, id.vars = c("threshold")) %>% filter(variable %in% c("accuracy", "p_high_threshold"))
  
  gg <- ggplot(res3, aes(x=threshold, y=value, colour=variable, group=variable)) + geom_line(lwd=2)  + 
    theme_light() +
    ggtitle(paste("Accuracy vs Model Threshold\nmodel: ", model,sep="")) + 
    xlab("Model Threshold") +
    ylab("Accuracy / Share (%)") +
    scale_x_continuous(limit = c(min(res3$threshold),1)) +
    scale_y_continuous(breaks = pretty(seq(min(res3$value),1,0.02))) +
    scale_color_brewer(type = "qual",palette = 6, guide =  guide_legend(title=NULL),  
                       labels = c("Accuracy (%)", "Proportion of Images Above Threshold (%)")) +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          legend.text = element_text(size=12),
          legend.position = "bottom",
          legend.box = "horizontal",
          legend.background = element_rect(size=1,colour="black"),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}




plot_threshold_vs_acc_class<- function(preds, model){
  
  
  
  # take only with most confidence and Threshold
  preds_max_p <- dplyr::group_by(preds, subject_id) %>% summarise(max_p=max(p)) 
  preds_0 <- left_join(preds, preds_max_p, by="subject_id") %>% filter(p==max_p) %>% dplyr::arrange(subject_id)
  # only take one image
  preds_min_image_id <- group_by(preds_0, subject_id) %>% summarise(image_id=min(image_id))
  preds_0 <- inner_join(preds_0,preds_min_image_id,by=c("subject_id","image_id"))
  res <- NULL

  
  # thresholds to test
  thresholds <- seq(min(preds$p),0.95,by=0.05)
  for (ii in seq_along(thresholds)){
    
    preds_1 <- preds_0 %>% filter(p>=thresholds[ii])
    head(preds_1)
    
    # accuracy per true class
    preds_class <- dplyr::group_by(preds_1,y_true) %>% summarise(matches = sum(y_true == y_pred), n = n()) %>%
      mutate(accuracy=matches/n)
    preds_class
    
    # get total class numbers and join
    class_numbers <- dplyr::group_by(preds_0, y_true) %>% summarise(n_total=n_distinct(subject_id))
    
    preds_class <- left_join(preds_class, class_numbers, by="y_true") %>% mutate(p_high_threshold=round(n/n_total,4))
    preds_class$threshold <- thresholds[ii]
    preds_class$p_high_threshold <- ifelse(preds_class$p_high_threshold>1,1,preds_class$p_high_threshold)
    
    res[[ii]] <- preds_class
  }
  res2 <- do.call(rbind,res)
  res3 <- melt(data = res2, id.vars = c("threshold", "y_true")) %>% filter(variable %in% c("accuracy", "p_high_threshold"))

  gg <- ggplot(res3, aes(x=threshold, y=value, colour=variable, group=variable)) + geom_line(lwd=2)  + 
    theme_light() +
    facet_wrap("y_true") +
    ggtitle(paste("Accuracy vs Model Threshold\nmodel: ", model,sep="")) + 
    xlab("Model Threshold") +
    ylab("Accuracy / Share (%)") +
    scale_x_continuous(limit = c(min(res3$threshold),1)) +
    scale_y_continuous(breaks = pretty(seq(min(res3$value),1,0.02))) +
    scale_color_brewer(type = "qual",palette = 6, guide =  guide_legend(title=NULL),  
                       labels = c("Accuracy (%)", "Proportion of Images Above Threshold (%)")) +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          legend.text = element_text(size=12),
          legend.position = "bottom",
          legend.box = "horizontal",
          legend.background = element_rect(size=1,colour="black"),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}


################################################ -
# Plot Confusion Matrix ----
################################################ -

plot_cm<- function(preds, model){
  # empty confusion matrix
  conf_empty <- expand.grid(levels(preds$y_true),levels(preds$y_true))
  names(conf_empty) <- c("y_true","y_pred")
  
  # accuracy per true class
  class_sum <- dplyr::group_by(preds,y_true) %>% summarise(n_class = n())
  conf <- dplyr::group_by(preds,y_true, y_pred) %>% summarise(n = n()) %>% left_join(class_sum) %>%
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
  return(gg)
}


################################################ -
# Distribution of predicted values
################################################ -

plot_dist_pred<- function(preds, model){
  preds_dist <- preds
  preds_dist$correct <- factor(ifelse(preds_dist$y_true!=preds_dist$y_pred,"Wrong","Correct"))
  gg <- ggplot(preds_dist, aes(x=p, fill=correct)) + geom_density(alpha=0.3) + 
    facet_wrap("y_true", scales="free") +
    ggtitle(paste("Predicted values - density distribution\nmodel: ", model,sep="")) + 
    theme_light() +
    xlab("Model Output") +
    ylab("Density") + 
    scale_fill_brewer(type = "qual",direction = 1, palette=2) +
    theme(axis.text = element_text(size=12),
          axis.title = element_text(size=14),
          legend.text = element_text(size=12),
          legend.position = "bottom",
          legend.box = "horizontal",
          legend.title = element_blank(),
          legend.background = element_rect(size=1,colour="black"),
          strip.text.x = element_text(size = 14, colour = "white", face="bold"))
  return(gg)
}


################################################ -
# Plot subject images
################################################ -

plot_subject_image <- function(preds, subjects, id, path_scratch, ii){


  preds <- fromJSON(paste("[",gsub(pattern = "'", "\"", x=as.character(preds[[1]])),"]",sep=""))
  preds <- melt(preds,value.name = "prob",variable.name = "class")
  sub <- subjects[id]
  url <- unlist(sub[[id]]['urls'])
  label <- unlist(sub[[id]]['label'])
  file_name <- paste(path_scratch,"image_",ii,".jpeg",sep="")
  download.file(url, destfile = file_name, mode = 'wb')
  
  img <- readJPEG(file_name)
  
  
  gg1 <- ggplot(data.frame(x=0:1,y= 0:1),aes(x=x,y=y), geom="blank") +
    annotation_custom(rasterGrob(img, width=unit(1,"npc"), height=unit(1,"npc")), 
                      -Inf, Inf, -Inf, Inf) + theme_minimal() +
    theme(axis.title = element_blank(), axis.text = element_blank()) +
    theme(plot.margin = unit(c(0.7,0.9,0,0.9), "cm"))
  
  # keep only top 5
  preds <- preds[order(preds$prob, decreasing = TRUE)[1:min(5,dim(preds)[1])],]
  
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
    theme(plot.margin = unit(c(0.7,0.9,0.5,0.9), "cm"), panel.border=element_rect(fill=NA)) +
    scale_y_continuous(expand=c(0,0), limits = c(0,1)) +
    labs(x=NULL)
  
  
  title=textGrob(label = label,gp=gpar(fontsize=20,fontface="bold"), vjust=1)
  
  gg <- arrangeGrob(gg1,gg2,top=title)
  
  rect <- grid.rect(.5,.5,width=unit(0.99,"npc"), height=unit(0.99,"npc"), 
              gp=gpar(lwd=3, fill=NA, col="black"))
  gg_comb <- list(gg, rect)


  return(gg_comb)
}