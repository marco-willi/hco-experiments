############################ -
# Libraries ----
############################ -

library(dplyr)
library(ggplot2)
library(reshape2)
library(jpeg)
library(grid)
library(gridExtra)
library(scales)

############################ -
# Load Functions ----
############################ -

source("analyses/plot_functions.R")

source("analyses/plot_parameters.R")

############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_project <- "db/camcat2/"
fname <- "classifications_experiment_20171208_exp_simulation.csv"
path_scratch <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/scratch/camcat2/"
path_save <- "D:/Studium_GD/Zooniverse/Project/Poster/CS_PosterFair_Fall2017/"
plot_path <- "D:/Studium_GD/Zooniverse/Results/publication/figures/"

############################ -
# Read Classifications ----
############################ -

data_raw <- read.csv2(paste(path_main, path_project, fname, sep=""),sep = ",", quote = "\"", header = TRUE)
head(data_raw)
names(data_raw)

table(data_raw$ret_retirement_reason, data_raw$X.experiment_group)
table(data_raw$ret_retirement_reason, data_raw$ret_workflow_id)





############################ -
# Categorize data ----
############################ -

retirement_flag <- function(ret_workflow_id, ret_retirement_reason, label_plur){
  ifelse(ret_retirement_reason != "" & ret_workflow_id == "4963.0",1,
      ifelse(grepl('Vehicle',label_plur) & ret_workflow_id == "5001.0",1,
           ifelse(grepl('Blank', label_plur) & ret_workflow_id == "5000.0",1,0)))
}


matcher <- function(y_exp, y_plur){
  match <- sapply(1:length(y_exp), function(x){
      grepl(pattern = y_exp[x], x = y_plur[x])
    })
  return(match)
}


species_allowed <- pretty_labels_conv(tolower(c("bird", "buffalo", "eland", "elephant", "gemsbock",
"giraffe", "HUMAN", "hyaenabrown",  "impala", "jackalblackbacked","baboon",
"kudu", "monkeybaboon", "rabbithare",  "rhino", "warthog",
"wildebeest", "zebra", "blank", "vehicle")))

# convert pluarlity labels
plur_levels <- levels(data_raw$label_plur)

plur_levels_new <- sapply(plur_levels, function(x){
  xx <- unlist(strsplit(as.character(x),","))
  xx2 <- sapply(xx, function(z){
    zz <- ifelse(grepl("'",z),gsub("'","",z),z)
    zz <- ifelse(grepl("\\[",zz),gsub("\\[","",zz),zz)
    zz <- ifelse(grepl("]",zz),gsub("]","",zz),zz)
    zz <- ifelse(grepl(" ",zz),gsub(" ","",zz),zz)
    zz <- pretty_labels_conv(zz)
    return(zz)
  })
  return(paste(xx2, collapse="#"))
})
plur_levels_new <-  sapply(plur_levels_new, function(x){gsub(pattern = "Novehicle", "Species", x)})
plur_levels_new = sapply(plur_levels_new, function(x){gsub(pattern = "Notblank", "Something", x)})

data <- data_raw
levels(data$label_plur) <- plur_levels_new
data <- select(data, X, subject_id, machine_label, machine_prob, retire_label_exp,
               label_plur, retirement_reason_plur, X.experiment_group, link, ret_workflow_id,ret_workflow_id, ret_retirement_reason) %>%
  mutate(machine_label=pretty_labels_conv(data_raw$machine_label),
         retire_label_exp=pretty_labels_conv(data_raw$retire_label_exp),
         retirement_reason_plur=pretty_labels_conv(retirement_reason_plur),
         retire_label_exp = ifelse(retire_label_exp == "", "Not_retired", as.character(retire_label_exp)),
         x_retired_real = retirement_flag(ret_workflow_id, ret_retirement_reason, label_plur),
         x_match_exp_plur = matcher(retire_label_exp, label_plur),
         x_match_mach_plur = matcher(machine_label, label_plur),
         x_different_outcome = ifelse(retire_label_exp != 'No_agreement' & !x_match_exp_plur,1,0),
         x_exp_eligible=ifelse(machine_label %in% c("Blank", "Vehicle"), 1, 
                               ifelse(machine_label %in% species_allowed & as.numeric(levels(machine_prob))[machine_prob] >= 0.85, 1, 0)),
         x_exp_eligible_strict=ifelse(machine_label %in% c(species_allowed,"Blank","Vehicle") & as.numeric(levels(machine_prob))[machine_prob] >= 0.95, 1, 0),
         c_res = ifelse(x_match_exp_plur & retirement_reason_plur != "Not retired","Exp_Plur_Match_Final",
                        ifelse(x_match_exp_plur & retirement_reason_plur == "Not retired", "Exp_Plur_Match_Preliminary",
                               ifelse(retire_label_exp == "No_agreement" & x_match_mach_plur & retirement_reason_plur != "Not retired", "No_Agreement_Mach_Match_Final",
                                      ifelse(retire_label_exp == "No_agreement" & x_match_mach_plur & retirement_reason_plur == "Not retired", "No_Agreement_Mach_Match_Preliminary",
                                        ifelse(retire_label_exp %in% c("No_agreement"),"No_Agreement_Fallback_Non_Experiment",
                                        ifelse(x_match_mach_plur & retirement_reason_plur == "Not retired", "Mach_Match_Preliminary",
                                                    ifelse((!retire_label_exp %in% c("No_agreement", "Not_retired", "")) & (retirement_reason_plur != "Not retired"),
                                                           "Different_Outcome","unfinished"))))))))

table(data$machine_label)
table(data$retirement_reason_plur)
table(data$label_plur)
  
table(data$x_exp_eligible, data$x_exp_eligible_strict)
prop.table(table(filter(data,x_exp_eligible==1)$c_res))

filter(data, c_res=="Exp_Plur_Match_Final")
filter(data, c_res =='Exp_Plur_Match_Preliminary')
filter(data, c_res =="No_Agreement_Mach_Match_Final")
filter(data, c_res =="No_Agreement_Mach_Match_Preliminary")
filter(data, c_res =="No_Agreement_Fallback_Non_Experiment")
filter(data, c_res =="Mach_Match_Preliminary")
filter(data, c_res =="Different_Outcome")
filter(data, c_res =="unfinished")

tt <- filter(data, machine_label %in% c("Jackalblackbacked", "Monkeybaboon", "Baboon")) 
table(droplevels(tt$machine_label),tt$c_res)

table(data$retirement_reason_plur, data$x_retired_real)
filter(data, x_retired_real==0 & retirement_reason_plur == "Nothing_here")

filter(data, x_retired_real==0 & retirement_reason_plur == "Classification_count")


# 1) Percentage Eligible
prop.table(table(data$x_exp_eligible))

# 2) Of Eligible - Percentage Retired 
tt <- filter(data,x_exp_eligible==1) 
prop.table(table(tt$x_retired_real))

# 3) Of Retired - Result Split Percentage
tt2 <- filter(tt, x_retired_real==1)
prop.table(table(tt2$c_res)) *100
paste(prop.table(table(tt2$c_res)) *100)

# 4) Of No Agreement - Final Agreements 
prop.table(table(filter(tt2, grepl(pattern = "No_Agree", c_res))$c_res))


# Samples - Different Outcome
head(filter(tt2, c_res=="Different_Outcome"))
tt3 <- filter(tt, x_retired_real==0)


# 5 ) Not yet Retired - Preliminary Results
prop.table(table(tt3$c_res)) * 100
filter(tt3, c_res=="Mach_Match_Preliminary")
filter(tt3, c_res=='No_Agreement_Fallback_Non_Experiment')



# 6) analyse different outcome
ttd <- filter(tt2, c_res=="Different_Outcome")
ttd
ttd <- dplyr::group_by(ttd, retire_label_exp, label_plur) %>% dplyr::summarise(n=n())
ttd
prop.table(ttd$n)
ggplot(ttd,aes(x=retire_label_exp, y=label_plur, fill=n)) + geom_tile()




# 7) Not Eligible - Percentage Retired 
tt <- filter(data,x_exp_eligible==0) 
prop.table(table(tt$x_retired_real))

# 3) Of Retired - Result Split Percentage
tt2 <- filter(tt, x_retired_real==1)
prop.table(table(tt2$c_res)) *100
paste(prop.table(table(tt2$c_res)) *100)

# 4) Of No Agreement - Final Agreements 
prop.table(table(filter(tt2, grepl(pattern = "No_Agree", c_res))$c_res))


# Samples - Different Outcome
head(filter(tt2, c_res=="Different_Outcome"))
tt3 <- filter(tt, x_retired_real==0)


# 5 ) Not yet Retired - Preliminary Results
prop.table(table(tt3$c_res)) * 100
filter(tt3, c_res=="Mach_Match_Preliminary")
filter(tt3, c_res=='No_Agreement_Fallback_Non_Experiment')



###################### -
# Per Class ----
###################### -

table(data$machine_label)

data_class <- dplyr::filter(data, X.experiment_group==0 & x_exp_eligible==1 & x_retired_real==1) %>% 
  dplyr::group_by(machine_label) %>% dplyr::summarise(p_correct_exp=sum(x_match_exp_plur)/n(),
                                        p_correct_mach=sum(x_match_mach_plur)/n(),
                                        p_unchanged=sum(x_different_outcome==0)/n())

head(data_class)
ggplot(data_class, aes(x=machine_label,y=p_correct_exp)) + geom_bar(stat="identity")

data_class_long <- melt(data_class,id.vars = "machine_label")
str(data_class_long)
levels(data_class_long$variable) <- c("ExpMatch","FinalMatch","Equal")
data_class_long$machine_label <- droplevels(data_class_long$machine_label)

ggplot(data_class_long, aes(x=machine_label,y=value * 100, fill=variable)) + 
  geom_bar(stat="identity", position="dodge") +
  theme_light() +
  xlab("Class") +
  coord_flip() +
  ylab("Share of Images (%)") +
  ggtitle("Experiment Outcome",subtitle = "" ) +
  geom_text(aes(label=paste("", sprintf("%.1f",round(value,3)*100)), y=102),
            size=text_normal, position = position_dodge(width = 1), vjust=0.1) +
  theme_small_plots +
  scale_y_continuous(limits=c(0,109), breaks = seq(0,100,by=25)) +
  scale_fill_brewer(type = "qual", palette = 2)



gg <- ggplot(data_class_long, aes(x=machine_label,y=value * 100, colour=variable, group=variable)) + 
  #geom_bar(stat="identity", position="dodge") +
  geom_line(stat="identity", lwd=1) +
  theme_light() +
  coord_flip() +
  xlab("") +
  ylab("Share of Images (%)") +
  ggtitle("Experiment Simulation",subtitle = "Different Outcomes per Class" ) +
  scale_y_continuous(limits=c(40,100), breaks = seq(40,100,by=20)) +
  scale_color_brewer(type = "qual", palette = 2) +
  theme_light() +
  coord_flip() +
  scale_x_discrete(limits=rev(levels(data_class_long$machine_label))) +
  # geom_text(aes(label=paste("", sprintf("%.1f",round(value,3)*100)), y=25),
  #           size=3, position = position_dodge(width = 1), vjust=0.5,hjust=-0.03) +
  theme(axis.text.x = element_text(size=text_large, angle = 0, hjust = 1, vjust=0.5),
        axis.title = element_text(size=text_large),
        axis.text.y = element_text(size=text_vlarge),
        strip.text.x = element_text(size = text_med, colour = "white", face="bold"),
        legend.title = element_blank(),
        text = element_text(size=text_vlarge), legend.position = "bottom")
gg
pdf(file = paste(plot_path,"results/exp_species_outcomes_vert.pdf",sep=""), width = fig_width_05c, height = fig_height_normal_plus)
plot(gg)
dev.off()




gg <- ggplot(data_class_long, aes(x=machine_label,y=value * 100, fill=variable)) + 
  geom_bar(stat="identity", position="dodge") +
  theme_light() +
  coord_flip() +
  xlab("") +
  ylab("Share of Images (%)") +
  ggtitle("Experiment Simulation",subtitle = "Different Outcomes per Class" ) +
  scale_y_continuous(limits = c(40,105), oob = rescale_none) +
  #scale_y_continuous(limits=c(40,100), breaks = seq(40,100,by=20)) +
  scale_fill_brewer(type = "qual", palette = 2) +
  theme_light() +
  coord_flip() +
  scale_x_discrete(limits=rev(levels(data_class_long$machine_label))) +
  geom_text(aes(label=paste("", sprintf("%.1f",round(value,3)*100))),
             size=2, position = position_dodge(width = 1), vjust=0.4,hjust=-0.0) +
  theme(axis.text.x = element_text(size=text_med, angle = 0, hjust = 1, vjust=0.5),
        axis.title = element_text(size=text_med),
        axis.text.y = element_text(size=text_large),
        strip.text.x = element_text(size = text_med, colour = "white", face="bold"),
        legend.title = element_blank(),
        text = element_text(size=text_med), legend.position = "bottom")
gg
pdf(file = paste(plot_path,"results/exp_species_outcomes_vert_bar.pdf",sep=""), width = fig_width_05c, height = fig_height_normal_plus)
plot(gg)
dev.off()


# Strict Analysis
tt <- filter(data,x_exp_eligible_strict==1) 
prop.table(table(tt$x_retired_real))
tt2 <- filter(tt, x_retired_real==1)
prop.table(table(tt2$c_res)) * 100
prop.table(table(filter(tt2, grepl(pattern = "No_Agree", c_res))$c_res))

tt3 <- filter(tt, x_retired_real==0)

prop.table(table(tt3$c_res)) * 100
filter(tt3, c_res=="Mach_Match_Preliminary")
filter(tt3, c_res=='No_Agreement_Fallback_Non_Experiment')








filter(tt2, c_res=="unfinished")

table(data$x_exp_eligible)
table(data$x_exp_eligible, data$X.experiment_group)
table(data$x_match_exp_plur, data$X.experiment_group, data$x_match_mach_plur)

tt <- filter(data, x_exp_eligible==1 & X.experiment_group==0)
str(tt)
tt[200:220,]


#filter(retire_label_exp != "" & retirement_reason_plur != 'Not Retired') %>% 

filter(data, retire_label_exp == "" & )
filter(data, x_different_outcome==1)


ggplot(data, aes(x=x_match_exp_plur, fill=factor(X.experiment_group))) + facet_grid(x_different_outcome~x_match_mach_plur, scales="free") + geom_bar(position="dodge")



tt <- filter(data, X.experiment_group==0 & x_exp_eligible==1)
str(tt)
table(tt$retire_label_exp)


head(tt)
table(tt$x_match_exp_plur)

tt2 <- filter(tt, retire_label_exp != "no_agreement")
table(tt2$x_different_outcome)

table(tt2$retirement_reason_plur)

table(tt2$retire_label_exp)

table(tt$machine_label)

tt3 <- filter(tt2, x_different_outcome==1)
table(tt3$retire_label_exp, tt3$label_plur)

hm <- group_by(tt3, retire_label_exp, label_plur) %>% summarise(n=n())
hm
gg <- ggplot(hm, aes(x=retire_label_exp, y=label_plur)) + 
  geom_tile(aes(fill = n), colour = "black") + theme_bw() +
  #scale_fill_gradient2(low="blue", mid="yellow", high="red", midpoint=0.5, guide =  FALSE) +
  scale_fill_gradient(low = "white", high = "steelblue", guide =  FALSE) + 
  #scale_fill_gradient2(low="white", mid="yellow", high="red", midpoint=0.5, guide =  FALSE) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
  xlab("Experiment Label") +
  ylab("Non-Experiment Label") +
  scale_x_discrete()
gg

hist(as.numeric(tt3$machine_prob))

# checks
filter(data_raw, X == '13128616')
