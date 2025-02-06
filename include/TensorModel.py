import tensorflow as tf
from tensorflow.keras import layers, models
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
    
    def create_cnn(self) -> tf.keras.Model:
        # Create a data augmentation layer
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
            tf.keras.layers.RandomRotation(0.1),        # Randomly rotate image up to 10%
            tf.keras.layers.RandomZoom(0.1)    # Randomly zoom image up to 10%
        ])
        
        # Create a model
        model = models.Sequential([
            data_augmentation,                                                      # Apply data augmentation
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),  # Input layer
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),                                   # Classification layer
            layers.Dense(10, activation="relu"),                                   # Classification layer to 10 output classes
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="categorial_crossentropy", metrics=["accuracy"])
        model.summary()

        return model