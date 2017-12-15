############################-
# Analyse Impact on number of
# classifications on aggregated
# plurality label 
############################-

############################ -
# Libraries ----
############################ -

library(dplyr)
library(reshape2)
library(stringr)
library(ggplot2)


############################ -
# Parameters ----
############################ -


input_path <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/db/elephant_expedition/ee_subject_to_label_mapping.csv"

data_raw <- read.csv2(input_path, sep = ",", header = TRUE, quote = "\"")

head(data_raw)


############################ -
# Analyse ----
############################ -

data <- data_raw %>% select(label, meta_data.plur_label_history, meta_data.n_users, subject_id)
head(data)

data$label <- data$label %>% str_replace_all("[[:punct:]]", "")

# split labels (ignore potential multi label annotations)
labels_clean <- data$meta_data.plur_label_history %>% str_replace_all("[[:punct:]]", "")
head(labels_clean)
splitted <- sapply(labels_clean, function(x){strsplit(x, " ")})
head(splitted)

# create long data frame
data_long <- data.frame(label = rep(data$label, sapply(splitted, length)), plur_label = unlist(splitted), seq = unlist(sapply(splitted, function(x) {seq(1,length(x),1)})),
                        subject_id = rep(data$subject_id, sapply(splitted, length)))
head(data_long)

data_long$plur_label[data_long$plur_label == "VEGETATIONNOANIMAL"] <-"blank"
data_long$label[data_long$label == "VEGETATIONNOANIMAL"] <-"blank"

############################ -
# Figures ----
############################ -

summary(data_long$seq)


data_long2 <- filter(data_long, seq<=10)
data_long2$x_change <- ifelse((data_long2$plur_label != lag(data_long2$plur_label)) & (lag(data_long2$subject_id) == data_long2$subject_id), 1 ,0)
data_long2$x_change[is.na(data_long2$x_change)] <- 0


data_agg <- group_by(data_long2, label) %>% summarise(n_tot_label=n_distinct(subject_id)) %>%
  left_join(group_by(data_long2, label, seq) %>% summarise(n_agree = sum(as.character(plur_label)==as.character(label)), n_change=sum(x_change)), by="label") %>%
  mutate(p_agree=n_agree / n_tot_label, p_change=n_change / n_tot_label)

head(data_agg)

ggplot(data_agg, aes(y=p_agree, x=seq)) + geom_line() + facet_wrap("label")


gg <- ggplot(data_agg, aes(y=p_change*100, x=factor(seq))) + geom_bar(stat="identity") + facet_wrap("label", scales="free") + theme_minimal() + 
  xlab("Annotation Number") + ylab("Share of Annotations that change final label (%)") +
  geom_text(aes(label=round(p_change,3)*100), size=4, colour="white", hjust=0.5,vjust=1) +
  ggtitle("Elephant Expedition - Annotation Analysis")
gg

pdf(file = "D:/Studium_GD/Zooniverse/Results/misc_analysis/ee_annotation_analysis.pdf", width=12, height=8)
gg
dev.off()


dat_agg3 <- group_by(data_long, label, seq) %>% summarise(n_per_seq=n())
head(dat_agg3)
ggplot(dat_agg3, aes(x=seq,y=n_per_seq)) + geom_line() + facet_wrap("label", scales="free") + scale_x_continuous(limits=c(0,10))
