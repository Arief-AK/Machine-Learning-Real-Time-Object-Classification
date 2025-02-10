from include.Logger import Logger
from include.Visualiser import Visualiser
from include.TensorModel import TensorModel

NUM_EPOCHS = 20

def create_predicition_matrix(model_handler: TensorModel, visualiser: Visualiser, model, x_test, y_test, str_model):
    conf_matrix = model_handler.compute_confusion_matrix(model, x_test, y_test)
    visualiser.plot_confusion_matrix(conf_matrix, model_handler.get_class_names(), str_model)

    # Produce diagonal-only confusion matrix
    diagonal_matrix = model_handler.get_diagonal_confusion_matrix(conf_matrix)
    visualiser.plot_diagonal_confusion_matrix(diagonal_matrix, model_handler.get_class_names(), str_model)

def train_base_model(model_handler: TensorModel, visualiser: Visualiser, logger: Logger, x_train, y_train, x_test, y_test) -> tuple:
    model_name = "base_model"
    model = model_handler.create_cnn()
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=64, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info(f"Model accuracy: {test_acc * 100:.2f}%")
    visualiser.plot_training_history(history, model_name)
    return (model, model_name), (test_acc)

def train_model_batch_normalised(model_handler: TensorModel, visualiser: Visualiser, logger: Logger, x_train, y_train, x_test, y_test) -> tuple:
    model_name = "batch_norm_model"
    model = model_handler.create_cnn(batch_normalisation=True)
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=64, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info(f"Model accuracy: {test_acc * 100:.2f}%")
    visualiser.plot_training_history(history, model_name)
    return (model, model_name), (test_acc)

if __name__ == "__main__":
    # Initialise variables
    model_acc_results = {}
    
    # Create a logger
    logger = Logger(__name__)

    # Initialise visualiser
    visualiser = Visualiser()

    # Initalise model handler
    model_handler = TensorModel()
    (x_train, y_train), (x_test, y_test) = model_handler.load_data()

    # Visualise sample data
    data_augmentation = model_handler.get_augmentation()
    visualiser.visualise_sample_images(5, x_train, 5, data_augmentation)

    # Create and train the base model
    (model, model_name), (test_acc) = train_base_model(model_handler, visualiser, logger, x_train, y_train, x_test, y_test)
    create_predicition_matrix(model_handler, visualiser, model, x_test, y_test, model_name)
    model_acc_results.update({model_name:test_acc})

    # Create and train the batch normalised model
    (model, model_name), (test_acc) = train_model_batch_normalised(model_handler, visualiser, logger, x_train, y_train, x_test, y_test)
    create_predicition_matrix(model_handler, visualiser, model, x_test, y_test, model_name)
    model_acc_results.update({model_name:test_acc})

    # Summarise the accuracy results of the models
    logger.info("Model Accuracy Results:")
    for model_name, accuracy in model_acc_results.items():
        logger.info(f"{model_name}:{accuracy * 100:.2f}%")

    logger.info("Done!")