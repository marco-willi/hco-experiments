#!/bin/bash

# process all files
python prep_season.py 1 ~/data/S1_data.csv
python prep_season.py 2 ~/data/S2_data.csv
python prep_season.py 3 ~/data/S3_data.csv
python prep_season.py 4 ~/data/S4_data.csv
python prep_season.py 5 ~/data/S5_data.csv
python prep_season.py 6 ~/data/S6_data.csv
python prep_season.py 7 ~/data/S7_data.csv
python prep_season.py 8 ~/data/S8_data.csv
python prep_season.py 9 ~/data/S9_data.csv
python prep_season.py s ~/data/season_data.csv