import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from include.Logger import Logger

class TensorModel:
    def __init__(self):
        self.logger = Logger(__name__)

    def load_data(self) -> tuple:
        # Load the CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalise the pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Convert class labels to one-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Log the dataset shape
        self.logger.debug(f"Training set: {x_train.shape}, Labels: {y_train.shape}")
        self.logger.debug(f"Test set: {x_test.shape}, Labels: {y_test.shape}")
    