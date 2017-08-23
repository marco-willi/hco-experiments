
############################ -
# Find some missclassifications ----
############################ -

random_wrongs <- filter(preds,correct==0)
set.seed(23)
random_wrongs <- random_wrongs[sample(size = 10,x=dim(random_wrongs)[1]),]
random_wrongs


############################ -
# Plot missclassifications ----
############################ -

# grobs <- list()
for (ii in 1:10){
  
  id <- paste(random_wrongs[ii,"subject_id"])
  preds <- random_wrongs[ii,"preds_all"]
  
  gg <- plot_subject_image(preds, subjects, id, path_scratch, ii)
  
  
  
  print_name = paste(path_figures,model,"_sample_wrong_",ii,sep="")
  pdf(file = paste(print_name,".pdf",sep=""), height=8, width=8)
  #grid.arrange(gg1,gg2,top=title)
  # grobs <- c(grobs,gg[[1]])
 
  grid.draw(gg[[1]])
  grid.draw(gg[[2]])
  grid.arrange(c(gg[[1]],gg[[2]]),newpage=FALSE)
  #do.call(grid.draw, gg)
  dev.off()
  png(file = paste(print_name,".png",sep=""), width=18, height=18,units = "cm", res=128)
  #grid.arrange(gg1,gg2,top=title)
  grid.draw(gg[[1]])
  grid.draw(gg[[2]])
  #do.call(grid.draw, gg)
  dev.off()
}
# gg_final <- arrangeGrob(grobs)
# gg_final
