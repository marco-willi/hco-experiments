""" Code to download raw images for classification """
from tools.predictor_external import PredictorExternal
from config.config import cfg_path


def predict_images():
    # score images
    model_files = [cfg_path['save'] +
                   'cc_species_v2_201708210308.hdf5',
                   cfg_path['save'] +
                   'cc_blank_vehicle_species_v2_201708200608.hdf5']

    pred_path = cfg_path['images'] + 'exp_south_africa_4'
    output_path = cfg_path['save']

    model_cfg_jsons = [cfg_path['save'] +
                       'cc_species_v2_201708210308_cfg.json',
                       cfg_path['save'] +
                       'cc_blank_vehicle_species_v2_201708200608_cfg.json']

    out_file_names = ['predictions_species_v2.csv', 'predictions_bvs_v2.csv']

    for model_file, model_cfg_json, output_file_name in \
     zip(model_files, model_cfg_jsons, out_file_names):

        # predictor
        predictor = PredictorExternal(
            path_to_model=model_file,
            model_cfg_json=model_cfg_json)

        # predict images in path
        predictor.predict_path(path=pred_path, output_path=output_path,
                               output_file_name=output_file_name)

if __name__ == "__main__":
    predict_images()
