from include.Logger import Logger
from include.Visualiser import Visualiser
from include.TensorModel import TensorModel

NUM_EPOCHS = 20

if __name__ == "__main__":
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

    # Create and train the model
    model = model_handler.create_cnn()
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=64, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test)

    logger.info(f"Model accuracy: {test_acc * 100:.2f}%")
    visualiser.plot_training_history(history)