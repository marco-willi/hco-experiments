# take a few lines from a csv
import csv

path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/project_data/\
elephant_expedition/"
path_file_in = "elephant-expedition-classifications.csv"
path_file_out = "elephant-expedition-classifications_small.csv"


path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/project_data/\
colorado_corridors/"

file_name_in = "colorado-corridors-project-classifications.csv"
file_name_out = "colorado-corridors-project-classifications_small.csv"


path = cfg_path['db']
file_name_in = 'classifications.csv'
file_name_out =  'classifications_sample.csv'


file_in = open(path + file_name_in , "r")
file_out = open(path + file_name_out, "w", newline='')
file_in_reader = csv.reader(file_in)
file_out_writer = csv.writer(file_out)
counter = 0
write_lines = 10000
for row in file_in_reader:
    file_out_writer.writerow(row)
    counter += 1
    if counter > write_lines:
        break
file_in.close()
file_out.close()
