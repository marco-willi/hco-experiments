############################ -
# Libraries ----
############################ -

library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(jpeg)
library(grid)
library(gridExtra)

############################ -
# Parameters Fix ----
############################ -

path_main <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/"
path_project <- "db/camcat2/"
fname <- "classifications_experiment_20171111_converted.csv"
path_scratch <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/scratch/camcat2/"
path_save <- "D:/Studium_GD/Zooniverse/Project/Poster/CS_PosterFair_Fall2017/"

############################ -
# Load Functions ----
############################ -

source("analyses/plot_functions.R")

source("analyses/plot_parameters.R")

############################ -
# Read Classifications ----
############################ -

data_raw <- read.csv2(paste(path_main, path_project, fname, sep=""),sep = ",", quote = "\"", header = TRUE)


############################ -
# Group to subject ----
############################ -


names(data_raw)[names(data_raw)=="workflow"] = "workflow_current"
data <- dcast(data = filter(data_raw, !attr %in% c("subject_id")), formula = subject_id + workflow_current ~ attr, value ="value")
names(data) <- sapply(names(data), function(x){gsub(pattern = "#","", x)})
data$machine_probability <- as.numeric(data$machine_probability)
data$machine_prediction <- pretty_labels_conv(data$machine_prediction)
data$plur_label <- pretty_labels_conv(data$plur_label)
data$workflow_current <- factor(data$workflow_current, levels=c("5000", "5001", "4963"))
data$plur_label <- ifelse(data$plur_label=="Notblank","Something",
                          ifelse(data$plur_label=="Notblank","Species", data$plur_label))

# group totals
group_totals <- group_by(data, experiment_group) %>% summarise(n_in_group=n_distinct(subject_id))
group_totals

data <- left_join(data, group_totals, by="experiment_group")

data$machine_prediction_workflow <- ifelse(data$workflow_current == '5000',
                                         ifelse(data$machine_prediction == "Blank", "Blank", "Something"),
                                         ifelse(data$workflow_current == '5001', 
                                                ifelse(data$machine_prediction == "Vehicle", "Vehicle", "Species"),data$machine_prediction))
data$x_pred_agreement <- ifelse(data$machine_prediction_workflow == data$plur_label,1,0)


########################### -
# Number of Annotations ----
########################### -

# total species covered
tt <- filter(data, (!machine_prediction %in% c("Vehicle","Blank")) & (experiment_group %in% c(1,2)))
prop.table(table(tt$experiment_group))


species_allowed <- pretty_labels_conv(tolower(c("bird", "buffalo", "eland", "elephant", "gemsbock",
                                                "giraffe", "HUMAN", "hyaenabrown",  "impala", "jackalblackbacked","baboon",
                                                "kudu", "monkeybaboon", "rabbithare",  "rhino", "warthog",
                                                "wildebeest", "zebra", "blank", "vehicle")))

data_ann <- filter(data_raw, attr %in% c("n_users","#experiment_group","#machine_prediction","#machine_probability"))
data_ann <- dcast(data_ann, formula = subject_id + workflow_current ~attr, value="value")
names(data_ann) <- sapply(names(data_ann), function(x){gsub(pattern = "#","", x)})
data_ann$machine_probability <- as.numeric(data_ann$machine_probability)
data_ann$machine_prediction <- pretty_labels_conv(data_ann$machine_prediction)


data_ann <- dplyr::group_by(data_ann, subject_id) %>% summarise(n_annotations=sum(as.numeric(n_users)),
                                                                experiment_group = max(experiment_group),
                                                                x_exp_eligible=max(ifelse(machine_prediction %in% c("Blank", "Vehicle"), 1, 
                                                                                      ifelse(machine_prediction %in% species_allowed & machine_probability >= 0.85, 1, 0))))
# check eligiblity flag
table(data_ann$experiment_group, data_ann$x_exp_eligible)

# Comparison Experiment vs Non-Experiment on eligible only
data_el <- filter(data_ann, x_exp_eligible==1)
tab <- group_by(data_el, experiment_group) %>% summarise(n_annotations=sum(n_annotations))
table(data_el$experiment_group)

# less annotations required on eligible subjects
1-(tab$n_annotations[2]/ tab$n_annotations[1])


tt <- group_by(data_ann, experiment_group) %>% summarise(n_annotations=sum(n_annotations))
c(tt$n_annotations[1], sum(tt$n_annotations[2:3]))

# less annotations required total
1-(sum(tt$n_annotations[2:3])/ tt$n_annotations[1])

# experiment group vs non-experiment annotations
(1/tt$n_annotations[1]) * sum(tt$n_annotations[2:3])







############################ -
# Visualize ----
############################ -

               
data_gg <- dplyr::group_by(data, workflow_current, experiment_group, retirement_reason) %>% dplyr::summarise(n=n()/max(n_in_group))
ggplot(data_gg, aes(x=workflow_current,y=n)) + geom_bar(stat="identity") + facet_grid(experiment_group~retirement_reason)


data_gg <- group_by(data, workflow_current, experiment_group) %>% summarise(n=n()/max(n_in_group))
ggplot(data_gg, aes(x=workflow_current,y=n)) + geom_bar(stat="identity") + facet_wrap("experiment_group")


# overlap
data_gg <- group_by(data, workflow_current, experiment_group, retirement_reason, n_users) %>% summarise(n=n()/max(n_in_group))
ggplot(data_gg, aes(x=workflow_current,y=n)) + geom_bar(stat="identity") + facet_grid(experiment_group~retirement_reason+n_users)


############################ -
# Inspect ----
############################ -

insp <- filter(data, workflow_current == "5000" & retirement_reason == "consensus" & n_users == 2 & experiment_group==1)
head(insp)


############################ -
# Check machine prediction
# and retirement ----
############################ -

# all current estimates
tt2 = group_by(data, workflow_current, machine_prediction_workflow, plur_label) %>% summarise(n=n())
n_total <- group_by(tt2, workflow_current) %>% summarise(n_workflow=sum(n))
tt3 <- left_join(tt2,n_total) %>% mutate(p_total=n/n_workflow)

ggplot(tt3, aes(x=machine_prediction_workflow,y=plur_label)) + 
  geom_tile(aes(fill = p_total), colour = "white") + 
  geom_text(aes(label = round(p_total, 3))) +
  scale_fill_gradient(low = "white",high = "steelblue") +
  facet_wrap("workflow_current", scales = "free")

# only retired subjects
tt2 = filter(data,  !is.na(ret_retired_at) & !(plur_label == "unknown answer label")) %>% group_by(workflow_current, machine_prediction_workflow, plur_label) %>% summarise(n=n())
n_total <- group_by(tt2, workflow_current) %>% summarise(n_workflow=sum(n))
tt3 <- left_join(tt2,n_total) %>% mutate(p_total=n/n_workflow)

gg <- ggplot(tt3, aes(x=machine_prediction_workflow,y=plur_label)) + 
  geom_tile(aes(fill = p_total), colour = "white") + 
  geom_text(aes(label = round(p_total, 3))) +
  scale_fill_gradient(low = "white",high = "steelblue") +
  facet_wrap("workflow_current", scales = "free") +
  theme_light() + theme(axis.text.x = element_text(angle = 90, hjust = 1))
gg

pdf(file = paste(path_save,"camcat_heat_map.pdf",sep=""),width = 14,height = 6)
gg
dev.off()


# look at wrong predictions
wrong_p <- filter(data, (tolower(machine_prediction_workflow) != tolower(plur_label)) & !is.na(ret_retired_at))
wrong_p[1:30,]


############################ -
# Check machine disagreement
# samples ----
############################ -

wrongs <- filter(data, (x_pred_agreement==0) & (experiment_group==1))

set.seed(456)
sample_wrongs <- wrongs[sample(1:dim(wrongs)[1],size = 6, replace = FALSE),]

urls <- sample_wrongs$link
preds <- sample_wrongs$machine_prediction_workflow
trues <- sample_wrongs$plur_label
probs <- sample_wrongs$machine_probability
subject_ids <- sample_wrongs$subject_id

# list of all figures
grobs <- list()
ncol=2
nrow=3
for (ii in 1:length(urls)){
  id <- subject_ids[ii]
  pred <- preds[ii]
  true <- trues[ii]
  prob <- probs[ii]
  url <- urls[ii]
  file_name <- paste(path_scratch,"image_",ii,".jpeg",sep="")
  download.file(url, destfile = file_name, mode = 'wb')
  
  img <- readJPEG(file_name)
  
  
  gg1 <- ggplot(data.frame(x=0:1,y= 0:1),aes(x=x,y=y), geom="blank") +
    annotation_custom(rasterGrob(img, width=unit(1,"npc"), height=unit(1,"npc")), 
                      -Inf, Inf, -Inf, Inf) + theme_minimal() +
    theme(axis.title = element_blank(), axis.text = element_blank()) +
    theme(plot.margin = unit(c(0.7,0.9,0,0.9), "cm"))
  
  pred <- data.frame(prob=prob, class=pred)
  
  gg2 <- ggplot(pred, aes(x=reorder(class, prob),y=prob)) + geom_bar(stat="identity") +
    coord_flip() +
    theme_light() +
    ylab("Model Output") +
    xlab("") +
    ggtitle("Model Prediction") +
    theme(axis.text.y=element_blank(), axis.text.x=element_text(size=16),
          axis.title.x=element_text(size=16),
          axis.title.y=element_text(size=16),
          plot.title = element_text(size=16),
          axis.ticks.y = element_blank()) +
    geom_text(aes(label=paste(class," (",round(prob,3)*100," %)",sep=""), y=0.05), size=4,fontface="bold", vjust="middle", hjust="left") +
    theme(plot.margin = unit(c(0.7,0.9,0.5,0.9), "cm"), panel.border=element_rect(fill=NA)) +
    scale_y_continuous(expand=c(0,0), limits = c(0,1)) +
    labs(x=NULL)
  title=textGrob(label = paste("True class: ",true,sep=""),gp=gpar(fontface="bold"), vjust=1)
  ga <- arrangeGrob(gg1,gg2,top=title)
  
  gb <- rectGrob(height = .98, width = .98, gp = gpar(lwd = 1.5, col = "blue",  fill=rgb(1, 1, 1, 0))) # border
  gt <- gTree(children = gList(ga, gb))
  
  grobs[[ii]] <- gt
}
ml <- marrangeGrob(grobs, nrow=nrow, ncol=ncol, top="")
ml




############################ -
# Check machine agreement
# and score distr ----
############################ -

# compare density distributions
text_large=10
text_med=8
text_small=6
ggplot(data, aes(x=machine_probability,fill=factor(x_pred_agreement))) + geom_density(alpha=0.3) +
  facet_wrap("workflow_current", scales="free") +
  ggtitle("Density of correctly/wrongly predicted classes") + 
  theme_light() +
  xlab("Model Output") +
  ylab("Density") + 
  scale_fill_brewer(type = "qual",direction = -1, palette=2) +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        legend.text = element_text(size=text_med),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_blank(),
        legend.background = element_rect(size=1,colour="black"),
        strip.text.x = element_text(size = text_small, colour = "white", face="bold")) +
  scale_y_sqrt()


# compare density distributions by experiment group
text_large=10
text_med=8
text_small=6
ggplot(data, aes(x=machine_probability,fill=factor(x_pred_agreement))) + geom_density(alpha=0.3) +
  facet_grid(workflow_current~experiment_group, scales="free") +
  ggtitle("Density of correctly/wrongly predicted classes") + 
  theme_light() +
  xlab("Model Output") +
  ylab("Density") + 
  scale_fill_brewer(type = "qual",direction = -1, palette=2) +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        legend.text = element_text(size=text_med),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_blank(),
        legend.background = element_rect(size=1,colour="black"),
        strip.text.x = element_text(size = text_small, colour = "white", face="bold")) +
  scale_y_sqrt()


# compare absolute values
ggplot(data, aes(x=machine_probability,fill=factor(x_pred_agreement))) + geom_histogram(position = "dodge", binwidth = 0.1) +
  facet_wrap("workflow_current", scales="free") +
  ggtitle("Density of correctly/wrongly predicted classes") + 
  theme_light() +
  xlab("Model Output") +
  ylab("# cases") + 
  scale_fill_brewer(type = "qual",direction = -1, palette=2) +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        legend.text = element_text(size=text_med),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_blank(),
        legend.background = element_rect(size=1,colour="black"),
        strip.text.x = element_text(size = text_small, colour = "white", face="bold"))

# compare absolute values by experiment group
ggplot(data, aes(x=machine_probability,fill=factor(x_pred_agreement))) + geom_histogram(position = "dodge", binwidth = 0.05) +
  facet_grid(workflow_current~experiment_group, scales="free") +
  ggtitle("Density of correctly/wrongly predicted classes") + 
  theme_light() +
  xlab("Model Output") +
  ylab("# cases") + 
  scale_fill_brewer(type = "qual",direction = -1, palette=2) +
  theme(axis.text = element_text(size=text_large),
        axis.title = element_text(size=text_large),
        legend.text = element_text(size=text_med),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.title = element_blank(),
        legend.background = element_rect(size=1,colour="black"),
        strip.text.x = element_text(size = text_small, colour = "white", face="bold"))

############################ -
# Look at retirement over time ----
############################ -

# retirement per day
ggplot(filter(data,!is.na(ret_retired_at)), aes(x=as.Date(ret_retired_at),fill=experiment_group)) + 
  geom_bar(position="dodge") + 
  facet_wrap("workflow_current", scales="free") +
  theme_light() +
  xlab("Date") +
  ylab("Retired Subjects")

# classifications per day
dat_gg <- filter(data, !is.na(ret_retired_at)) %>% 
  group_by(as.Date(ret_retired_at), workflow_current, experiment_group) %>%
  summarise(n_users=sum(as.numeric(n_users)), n_ret=n())
names(dat_gg) <- c("ret_retired_at", "workflow_current", "experiment_group", "n_users", "n_ret")
ggplot(dat_gg, aes(x=ret_retired_at,y=n_users/n_ret, colour=experiment_group)) + 
  geom_line(lwd=2)+ 
  facet_grid(workflow_current~.) +
  theme_light() +
  scale_y_continuous(limits=c(0,10)) +
  xlab("Date") +
  ylab("Average Annotations / Retirement")





