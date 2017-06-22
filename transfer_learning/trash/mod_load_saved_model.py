from confi.config import cfg_model
from keras.models import load_model

def load_model(cfg_model=cfg_model):

    model = load_model(cfg_model['load_model'] + '.hdf5')

    return model