""" Code to import and process predictions """
import pandas as pd
from config.config import cfg_path
import random


# Parameters
pred_species_file = cfg_path['save'] + 'predictions_species.csv'
pred_vbs_file = cfg_path['save'] + 'predictions_vehicle_blank_species.csv'
manifest_file = cfg_path['db'] + 'manifest_set_28_2017.09.07.csv'
output_manifest = cfg_path['db'] + 'NEW_manifest_set_28_2017.09.07.csv'
output_full = cfg_path['db'] + 'CHECK_manifest_set_28_2017.09.07.csv'

# import files
pred_species = pd.read_csv(pred_species_file)
pred_vbs = pd.read_csv(pred_vbs_file)

pred_species.head
pred_species.columns

pred_vbs.head
pred_vbs.columns

# re-arrange columns
pred_vbs = pred_vbs[pred_species.columns]
assert all(pred_vbs.columns == pred_species.columns)
assert all(pred_vbs['file_name'] == pred_species['file_name'])

# determine final class of each observation
final_prediction = pred_vbs['predicted_class'].copy()
species_preds_index = final_prediction.index[final_prediction == 'species']

# build final data frame
pred_final = pred_vbs.copy()
pred_final.iloc[species_preds_index, :] = \
    pred_species.iloc[species_preds_index, :]

pred_final[0:4]


# build original columns to match with manifest
pred_final['subject_id'] = [int(x.split('_')[1])
                            for x in pred_final['file_name']]
pred_final['id_order'] = [x.split('_')[0]
                          for x in pred_final['file_name']]
pred_final['image_name'] = [x[x.find('_', x.find('_') + 1)+1:]
                            for x in pred_final['file_name']]

# create additional columns
pred_final = pred_final.rename(columns={
    'predicted_probability': '#machine_probability',
    'predicted_class': '#machine_prediction'})

pred_final[pred_final['subject_id'] == 1]

# read manifest
manifest = pd.read_csv(manifest_file)
manifest[manifest['subject_id'] == 1]
manifest.columns
pred_final.columns

# join
manifest_merge = pd.merge(manifest, pred_final,
                          how="inner", on=['subject_id', 'image_name'])

# create experiment group
idx = [x for x in range(0, manifest_merge.shape[0])]

# randomly assign 50% to experiment
random.seed(233)
idx_exp = random.sample(idx, k=int(len(idx)/2))
manifest_merge['#experiment_group'] = 0
manifest_merge.loc[idx_exp, '#experiment_group'] = 1

# remove species preds that have too low confidence and are not in pre-defined
# species
threshold = 0.85
species_allowed = ["bird", "buffalo", "eland", "elephant", "gemsbock",
                   "giraffe", "HUMAN", "hyaenabrown",  "impala", "jackal",
                   "kudu", "monkeybaboon", "rabbithare",  "rhino", "warthog",
                    "wildebeest", "zebra", "blank", "vehicle"]

idx_blank = \
   (manifest_merge['#experiment_group'] == 1) & \
     (((manifest_merge['#machine_probability'] < threshold) &
      (~manifest_merge['#machine_prediction'].isin(['blank', 'vehicle']))) |
     (~manifest_merge['#machine_prediction'].isin(species_allowed)))

manifest_merge.loc[idx_blank, '#experiment_group'] = 3


# Create final export
manifest_merge.to_csv(output_full, index=False)

final_columns = list(manifest.columns)
final_columns = final_columns + ['#experiment_group', '#machine_prediction',
                                 '#machine_probability']
manifest_final = manifest_merge[final_columns]

manifest_final.to_csv(output_manifest, index=False)
