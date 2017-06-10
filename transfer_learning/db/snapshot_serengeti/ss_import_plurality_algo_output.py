import os


# find relevant files
files_in_dir = os.listdir()
plur_files = list()

for f in files_in_dir:
    if 'plurality' in f:
        plur_files.append(f)


# read files and extract information
subject_zooniverse_id,capture_event_id,retire_reason,
season,site,roll,filenames,number_of_classifications,
number_of_votes,number_of_blanks,pielou_evenness,
number_of_species,species_index,species,species_votes,
species_fraction_support,species_count_min,species_count_median,
species_count_max,species_fraction_standing,species_fraction_resting,
species_fraction_moving,species_fraction_eating,
species_fraction_interacting,species_fraction_babies


ASG000xz5h,51e8a1c0e0053a09c3000003,consensus,S6,B03,R2,IMAG0001.JPG,
32,23,9,0,1,1,human,23,1.0,1,1,1,0.521739130435,0.0,0.0,0.0,0.478260869565,0.0