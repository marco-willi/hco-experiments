

############################ -
# Find some missclassifications ----
############################ -

random_wrongs <- filter(preds,correct==0)
set.seed(23)
random_wrongs <- random_wrongs[sample(size = 12,x=dim(random_wrongs)[1]),]
random_wrongs


gg <- plot_subject_image_set(pred_set=random_wrongs, subjects, path_scratch, n_samples=10, ncol=2, nrow=3)
gg

g12 <- cbind(gg[[1]], gg[[2]], size="first")
g12$heights <- unit.pmax(gg[[1]][["heights"]], gg[[2]][["heights"]])
g34 <- cbind(gg[[3]], gg[[4]], size="first")
g34$heights <- unit.pmax(gg[[3]][["heights"]], gg[[4]][["heights"]])
g1234 <- rbind(g12, g34, size="first")
g1234$widths <- unit.pmax(g12[["widths"]], g34[["widths"]])
grid.newpage()
grid.draw(g1234)


############################ -
# Plot missclassifications ----
############################ -

grobs <- list()
for (ii in 1:10){
  
  id <- paste(random_wrongs[ii,"subject_id"])
  preds <- random_wrongs[ii,"preds_all"]
    
  
  preds0 <- fromJSON(paste("[",gsub(pattern = "'", "\"", x=as.character(preds[[1]])),"]",sep=""))
  preds0 <- melt(preds0,value.name = "prob",variable.name = "class")
  sub <- subjects[id]
  url <- unlist(sub[[id]]['urls'])[1]
  label <- unlist(sub[[id]]['label'])
  file_name <- paste(path_scratch,"image_",ii,".jpeg",sep="")
  download.file(url, destfile = file_name, mode = 'wb')
  
  img <- readJPEG(file_name)
  # img1 <-  rasterGrob(as.raster(img), interpolate = FALSE)
  # grid.draw(img1)
  
  gg1 <- ggplot(data.frame(x=0:1,y= 0:1),aes(x=x,y=y), geom="blank") +
    annotation_custom(rasterGrob(as.raster(img), width=unit(1,"npc"), height=unit(1,"npc"), interpolate = FALSE), 
                      -Inf, Inf, -Inf, Inf) + theme_minimal() +
    theme(axis.title = element_blank(), axis.text = element_blank()) +
    theme(plot.margin = unit(c(0.7,0.9,0,0.9), "cm"))
  gg1
  
  # keep only top 5
  preds0 <- preds0[order(preds0$prob, decreasing = TRUE)[1:min(5,dim(preds0)[1])],]
  
  # identify correct one and color differently
  hit_id <- which(preds0$class == label)
  colours <- rep("lightblue",dim(preds0)[1])
  # colours[hit_id] <- "salmon"
  colours[hit_id] <- "springgreen"
  
  gg2 <- ggplot(preds0, aes(x=reorder(class, prob),y=prob)) + geom_bar(stat="identity", fill=rev(colours)) +
    coord_flip() +
    theme_light() +
    ylab("Model Output") +
    xlab("") +
    ggtitle("Model Predictions (ordered)") +
    theme(axis.text.y=element_blank(), axis.text.x=element_text(size=16),
          axis.title.x=element_text(size=16),
          axis.title.y=element_text(size=16),
          plot.title = element_text(size=16),
          axis.ticks.y = element_blank()) +
    geom_text(aes(label=paste(class," (",round(prob,3)*100," %)",sep=""), y=0.05), size=5,fontface="bold", vjust="middle", hjust="left") +
    theme(plot.margin = unit(c(0.7,0.9,0.5,0.9), "cm"), panel.border=element_rect(fill=NA)) +
    scale_y_continuous(expand=c(0,0), limits = c(0,1)) +
    labs(x=NULL)
  
  font_size_adjuster <- function(cex, nchars){
    if (nchars<=10){
      return(cex)
    } else if (nchars<=15){
      return(cex*0.66)
    } else{
      return(cex*0.5)
    }
  }
  title=textGrob(label = paste("True class: ",label,sep=""),gp=gpar(fontsize=font_size_adjuster(20,nchar(label)),fontface="bold"), vjust=1)
  
  ga <- arrangeGrob(gg1,gg2,top=title)
  
  gb <- rectGrob(height = .98, width = .98, gp = gpar(lwd = 1.5, col = "blue", fill=rgb(1, 1, 1, 0))) # border
  gt <- gTree(children = gList(ga, gb))
  
  grobs[[ii]] <-gt
}
ml <- marrangeGrob(grobs, nrow=3, ncol=2, top="")
ml
