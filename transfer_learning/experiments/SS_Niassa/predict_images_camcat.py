""" Predict Images """
from tools.predictor_external import PredictorExternal


def predict_images():
    """ Predict Images in Directory """

    # class mapping according to model
    model_cfg_jsons = ['/host/data_hdd/save/camera_catalogue' +
                       'cc_species_v2_201708210308_cfg.json',
                       '/host/data_hdd/save/camera_catalogue' +
                       'cc_blank_vehicle_species_v2_201708200608_cfg.json']

    # score images
    model_files = ['/host/data_hdd/save/ss/ss_species_51_201708072308.hdf5',
                   '/host/data_hdd/models/ss/' +
                   'ss_blank_vs_non_blank_small_201707172207_model_best.hdf5']

    # input images for scoring
    pred_path = '/host/data_hdd/images/niassa/'

    # save csv with scored images
    output_path = '/host/data_hdd/save/niassa/'
    out_file_names = ['predictions_species_cc.csv', 'predictions_blank_cc.csv']

    for model_file, model_cfg_json, output_file_name in \
     zip(model_files, model_cfg_jsons, out_file_names):

        print("Starting with model %s" % model_file)

        # predictor
        predictor = PredictorExternal(
            path_to_model=model_file,
            model_cfg_json=model_cfg_json,
            refit_on_data=True)

        # predict images in path
        predictor.predict_path(path=pred_path, output_path=output_path,
                               output_file_name=output_file_name)


if __name__ == "__main__":
    predict_images()
