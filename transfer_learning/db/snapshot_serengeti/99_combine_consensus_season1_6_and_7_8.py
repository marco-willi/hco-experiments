# file to combine consensus annotations from season1-6 and season 7 and 8
# Module to get data stored at MSI at Dryad
from urllib import request
import csv
import numpy as np
from config.config import config, cfg_path


 # load config
db_path = cfg_path['db']

# consensus 1-6
output_file = db_path + 'consolidated_annotations_16.csv'

season7_8_file = db_path + 'consensus_7-8.csv'

new_file = db_path + 'consolidated_annotations.csv'

############################
# Get Data From Local Drive
############################

file_new = open(new_file, "w", newline='')
file_16 = open(output_file, "r")
file_78 = open(season7_8_file, "r")


new_writer = csv.writer(file_new, delimiter=',',
                                   quotechar='"',
                                   quoting=csv.QUOTE_ALL)

reader_16 = csv.reader(file_16)
reader_78 = csv.reader(file_78)

headers = next(reader_16)
headers2 = next(reader_78)

# map headers
idx = list()
for h in headers:
    if headers2.count(h) > 0:
        idx.append(headers2.index(h))
    else:
        idx.append(-1)

new_writer.writerow(headers)
for r in reader_16:
    new_writer.writerow(r)


for r in reader_78:
    r_new = list()
    for i in range(0, len(idx)):
        if idx[i] > -1:
            r_new.append(r[idx[i]])
        else:
            r_new.append('')
    new_writer.writerow(r_new)

file_new.close()
file_16.close()
file_78.close()


