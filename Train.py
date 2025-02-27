import os.path
import tensorflow as tf

from include.Visualiser import Visualiser
from include.Logger import Logger, logging
from include.TensorModel import TensorModel
from include.ModelProfiler import ModelProfiler

NUM_EPOCHS = 25
BATCH_FITTING = 128
BATCH_PROFILING = [32, 64, 128]

MODELS = ["base_model", "batch_norm_model", "batch_norm_model_sgd", "batch_norm_model_rmsprop"]
USE_EXISTING_MODELS = False

SAVE_MODELS = True
SAVE_MODELS_AS_H5 = True
SAVE_MODELS_AS_KERAS = True
SAVE_MODELS_AS_SavedModel = True

def create_predicition_matrix(model_handler: TensorModel, visualiser: Visualiser, model, x_test, y_test, str_model):
    conf_matrix = model_handler.compute_confusion_matrix(model, x_test, y_test)
    visualiser.plot_confusion_matrix(conf_matrix, model_handler.get_class_names(), str_model)

    # Produce diagonal-only confusion matrix
    diagonal_matrix = model_handler.get_diagonal_confusion_matrix(conf_matrix)
    visualiser.plot_diagonal_confusion_matrix(diagonal_matrix, model_handler.get_class_names(), str_model)

def train_model(model_name:str, model_handler: TensorModel, visualiser: Visualiser, logger: Logger, x_train, y_train, x_test, y_test, batch_size) -> tuple:
    # Check if model exists
    if USE_EXISTING_MODELS and os.path.exists(f"models/{model_name}.h5"):
        model = model = tf.keras.models.load_model(f"models/{model_name}.h5")
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.build()
        model.summary()
    else:
        if model_name == "base_model":
            model = model_handler.create_cnn()
        else:
            model = model_handler.create_cnn(batch_normalisation=True)

    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=batch_size, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)

    if SAVE_MODELS:
        if SAVE_MODELS_AS_H5:
            model.save(f"models/{model_name}.h5")

        if SAVE_MODELS_AS_KERAS:
            model.save(f"models/{model_name}.keras")

        if SAVE_MODELS_AS_SavedModel:
            model.export(f"models/{model_name}_saved_model")
    
    logger.info(f"Model accuracy: {test_acc * 100:.2f}%")
    visualiser.plot_training_history(history, model_name)
    return (model, model_name), (test_acc)

def train_models(model_handler: TensorModel, visualiser: Visualiser, logger: Logger, x_train, y_train, x_test, y_test, model_acc_results:dict) -> dict:
    for model_name in MODELS:
        (model, model_name), (test_acc) = train_model(model_name, model_handler, visualiser, logger, x_train, y_train, x_test, y_test, BATCH_FITTING)
        create_predicition_matrix(model_handler, visualiser, model, x_test, y_test, model_name)
        model_acc_results.update({model_name:test_acc})
    
    return model_acc_results

def profile_models(model_acc_results:dict, visualiser: Visualiser, logger: Logger):
    # Initialise profiler
    profiler = ModelProfiler()

    for batch_size in BATCH_PROFILING:
        # Load models
        for model_name, accuracy in model_acc_results.items():
            # Load the model and data
            model = model_handler.load_model(f"models/{model_name}.h5")
            (_, _), (x_test, _) = model_handler.load_data()
            (batch_time, throughput_time), (single_image_time) = profiler.measure_average_inference_time(batch_size, model, x_test, show_single_image_inference=True)

            # Plot the batch timings
            timings = [single_image_time, batch_time, throughput_time]
            visualiser.plot_inference_timings(timings, batch_size, model_name)
            
            # Log the results
            logger.info(f"Model: {model_name}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Batch time: {batch_time:.2f}ms")
            logger.info(f"Throughput: {throughput_time:.2f} images/s")
            logger.info(f"Accuracy:{accuracy * 100:.2f}%\n")

if __name__ == "__main__":
    # # Initialise variables
    model_acc_results = {}
    
    # # Create a logger
    logger = Logger(__name__)
    logger.set_level(logging.INFO)

    # # Initialise visualiser
    visualiser = Visualiser()

    # Initalise model handler
    model_handler = TensorModel()
    (x_train, y_train), (x_test, y_test) = model_handler.load_data()

    # Visualise sample data
    data_augmentation = model_handler.get_augmentation()
    visualiser.visualise_sample_images(5, x_train, 5, data_augmentation)

    # Train and profile models
    model_acc_results = train_models(model_handler, visualiser, logger, x_train, y_train, x_test, y_test, model_acc_results)
    profile_models(model_acc_results, visualiser, logger)
    
    logger.info("Done!")