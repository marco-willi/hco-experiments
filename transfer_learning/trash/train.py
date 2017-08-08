# model for handwritten-digits data on Zooniverse
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from config.config import config, cfg_path
import time
from learning.model_components import create_data_generators
from learning.model_components import model_save, model_param_loader
import importlib
from config.config import logging


def train(train_set, test_set, val_set):
    ##################################
    # Parameters + Config
    ##################################

    logging.info("Loading Parameters and Config")

    # general configs
    cfg = model_param_loader(config)

    # load model config
    mod_cfg = importlib.import_module('learning.config.' + cfg['model_config'])

    # load model file
    mod_file = importlib.import_module('learning.models.' + cfg['model_file'])

    ##################################
    # Data Generators
    ##################################

    logging.info("Creating Pre-Processing / Data Generators")

    train_generator, test_generator,\
        val_generator = mod_cfg.create_pre_processing(cfg, cfg_path)

    ##################################
    # Model Definition
    ##################################

    logging.info("Loading and compiling model")

    model = mod_file.build_model(cfg)

    opt = mod_cfg.create_model_optimizer()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    ##################################
    # Logging
    ##################################

    logging.info("Creating Logging modules")

    # save model weights after each epoch if training loss
    # decreases
    checkpointer = ModelCheckpoint(filepath=cfg_path['models'] +
                                   "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                   verbose=1,
                                   save_best_only=True)

    # log to csv
    csv_logger = CSVLogger(cfg_path['logs'] + 'training.log')

    # Tensorboard logger
    tb_logger = TensorBoard(log_dir=cfg_path['logs'], histogram_freq=0,
                                # batch_size=int(cfg['batch_size']),
                                write_graph=True
                                # write_grads=False, write_images=False,
                                # embeddings_freq=0,
                                # embeddings_layer_names=None,
                                # embeddings_metadata=None
                                )

    # add custom callbacks


    ##################################
    # Training
    ##################################

    time_s = time.time()
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n // cfg['batch_size'],
            epochs=cfg['num_epochs'],
            workers=4,
            pickle_safe=False,
            validation_data=test_generator,
            validation_steps=test_generator.n // cfg['batch_size'],
            callbacks=[checkpointer, csv_logger, tb_logger])

    print("Finished training after %s minutes" %
          ((time.time() - time_s) // 60))

    ##################################
    # Evaluation
    ##################################

    # Test Data
    eval_metrics = model.evaluate_generator(
                    test_generator,
                    steps=test_generator.n // cfg['batch_size'],
                    workers=4,
                    pickle_safe=False)

    # print evaluation
    print("Test Results")
    for name, value in zip(model.metrics_names, eval_metrics):
        print("%s: %s" % (name, value))

    # Validation Data
    eval_metrics = model.evaluate_generator(
                    val_generator,
                    steps=val_generator.n // cfg['batch_size'],
                    workers=4,
                    pickle_safe=False)

    # print evaluation
    print("Validation Results")
    for name, value in zip(model.metrics_names, eval_metrics):
        print("%s: %s" % (name, value))

    ##################################
    # Save
    ##################################

    model_save(model, config)
