# Model configuration for SS model
from models.helpers import create_data_generators
from keras.callbacks import LearningRateScheduler
from keras.optimizers import rmsprop, SGD

def create_callbacks(callback_list=[]):
    # learning rate function
    def lrnrate_dec(epoch):
        if epoch < 18:
            return 0.01
        elif epoch < 29:
            return 0.005
        elif epoch < 43:
            return 0.001
        elif epoch < 52:
            return 5e-4
        else:
            return 1e-4

    # learning rate decay rule
    learning_rate_decay = LearningRateScheduler(lrnrate_dec)

    callback_list.append(learning_rate_decay)
    return callback_list

def create_pre_processing(cfg, cfg_path):
    return create_data_generators(cfg, cfg_path, data_augmentation="ss")

def create_model_optimizer():
    # initiate RMSprop optimizer
    opt = SGD(lr=0.0001, decay=0)
    return opt

def get_class_mapping():
    pass