""" Code to download raw images for classification """
from tools.predictor_external import PredictorExternal
from config.config import cfg_path

# score images
model_file = cfg_path['save'] + 'cc_species_v2_201708210308.hdf5'
model_file = cfg_path['save'] + 'cc_blank_vehicle_species_v2_201708200608.hdf5'

pred_path = cfg_path['images'] + 'exp_south_africa_4'
output_path = cfg_path['save']
model_cfg_json = cfg_path['save'] + 'cc_species_v2_201708210308_cfg.json'
model_cfg_json = cfg_path['save'] + 'cc_blank_vehicle_species_v2_201708200608_cfg.json'

predictor = PredictorExternal(
    path_to_model=model_file,
    model_cfg_json=model_cfg_json)

predictor.predict_path(path=pred_path, output_path=output_path)
