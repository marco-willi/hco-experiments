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


input_path <- "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/db/elephant_expedition/result.csv"

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


############################ -
# Figures ----
############################ -

summary(data_long$seq)

data_long2 <- filter(data_long, seq<=10)

data_agg <- group_by(data_long2, label) %>% summarise(n_tot_label=n_distinct(subject_id)) %>%
  left_join(group_by(data_long2, label, seq) %>% summarise(n_agree = sum(as.character(plur_label)==as.character(label))), by="label") %>%
  mutate(p_agree=n_agree / n_tot_label)

head(data_agg)

ggplot(data_agg, aes(y=p_agree, x=seq)) + geom_line() + facet_wrap("label")
