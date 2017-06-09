############ Lucifer Cluster #################

def proc():
    # process raw classifications file
    import csv
    fname = 'raw_data.csv'
    path = '/home/will5448/data/' + fname
    path_write = '/home/will5448/data/'

    # header of csv file
    header = ['id', 'user_name', 'subject_zooniverse_id', 'capture_event_id',
     'created_at', 'retire_reason', 'season', 'site', 'roll',
     'filenames', 'timestamps', 'species', 'species_count',
     'standing', 'resting', 'moving', 'eating', 'interacting', 'babies']

    #header_new = header[0:10] + header[11:]
    header_new = header

    file = open(path, "r")
    datareader = csv.reader(file)

    # define seasons
    seasons = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', '10','WF1']
    season_files = dict()
    season_writers = dict()

    # create season files
    for s in seasons:
        season_files[s] = open(path_write + s + "_data.csv", "w")
        season_writers[s] = csv.writer(season_files[s], delimiter=',',
                                       quotechar='"',
                                       quoting=csv.QUOTE_ALL)
        # add header
        season_writers[s].writerow(header_new)

    count = 0
    for row in datareader:
        count += 1
        if count==1:
            continue
        # apply filters
        if ((row[3] == 'tutorial') & (row[6] not in ('S9','10','WF1'))) or \
           (row[6] == 'tutorial') or \
           (row[5] in ('blank', 'blank_consensus')):
            continue
        # remove additional column
        # del row[10]
        # write to file
        s = row[6]
        season_writers[s].writerow(row)

    # close all files
    for s in seasons:
        season_files[s].close()

    file.close()


if __name__ == '__main__':
    proc()




############ TESTS ###########################

# read snapshot serengeti raw data
import csv
import numpy as np

fname = '2017-06-04_serengeti_classifications.csv'
path = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/" + \
       "project_data/snapshot_serengeti/" + fname

path_write = "D:/Studium_GD/Zooniverse/Data/transfer_learning_project/\
project_data/snapshot_serengeti/"

# header of csv file
header = ['id', 'user_name', 'subject_zooniverse_id', 'capture_event_id',
 'created_at', 'retire_reason', 'season', 'site', 'roll',
 'filenames', 'timestamps', 'species', 'species_count',
 'standing', 'resting', 'moving', 'eating', 'interacting', 'babies']

header_new = header[0:10] + header[11:]
# seasons
#S1
#S2
#S3
#S4
#S5
#S6
#S7
#S8
#S9
#season
#tutorial


# prepare files as recommended
# - remove tutorial classifications
# - remove blank consensus
# - create season specific files


file = open(path, "r")
datareader = csv.reader(file)


# define seasons
seasons = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', '10','WF1']
season_files = dict()
season_writers = dict()

# create season files
for s in seasons:
    season_files[s] = open(path_write + s + "_data.csv", "w")
    season_writers[s] = csv.writer(season_files[s], delimiter=',',
                                   quotechar='"',
                                   quoting=csv.QUOTE_ALL)
    # add header
    season_writers[s].writerow(header_new)

count = 0
max_count = 30
for row in datareader:
    count += 1
    if count==1:
        continue
    # apply filters
    if ((row[3] == 'tutorial') & (row[6] not in ('S9'))) or \
       (row[6] == 'tutorial') or \
       (row[5] in ('blank', 'blank_consensus')):
        continue
    # remove additional column
    del row[10]
    # write to file
    s = row[6]
    season_writers[s].writerow(row)
    if count >= max_count:
        break

# close all files
for s in seasons:
    season_files[s].close()

file.close()




file = open(path, "r")
datareader = csv.reader(file)
count = 0
max_count=30
season_list = list()
retire_list = list()

for row in datareader:
    count += 1
    if count==1:
        continue
    season_list.append(row[6])
    retire_list.append(row[5])
    if (row[6] == 'S9') & ((count % 10000) == 0):
        print(row)
    if (row[6] == '10') & ((count % 10000) == 0):
        print(row)
    if (row[6] == 'WF1') & ((count % 10000) == 0):
        print(row)
    # apply filters
    if ((row[3] == 'tutorial') & (row[6] not in ('S9'))) or \
       (row[6] == 'tutorial') or \
       (row[5] in ('blank', 'blank_consensus')):
        continue


file.close()


season_counts = np.unique(season_list, return_counts=True)
retire_counts = np.unique(retire_list, return_counts=True)

for s, c in zip(season_counts[0], season_counts[1]):
    print("Season: %s, Counts: %s" % (s,c))

for s, c in zip(retire_counts[0], retire_counts[1]):
    print("Reason: %s, Counts: %s" % (s,c))


