import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import cifar10       # type: ignore
from tensorflow.keras.utils import to_categorical   # type: ignore

from sklearn.metrics import confusion_matrix

from include.Logger import Logger
from include.CNNBuilder import CNNBuilder

class TensorModel:
    def __init__(self):
        self.logger = Logger(__name__)

    def get_class_names(self) -> list:
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        return class_names
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model

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

        return (x_train, y_train), (x_test, y_test)
    
    def get_augmentation(self) -> tf.keras.Sequential:
        # Create a data augmentation layer
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),                                                               # Randomly flip images horizontally
            tf.keras.layers.RandomRotation(0.1),                                                                    # Randomly rotate image up to 10%
            tf.keras.layers.RandomZoom(0.1),                                                                        # Randomly zoom image up to 10%
            tf.keras.layers.RandomContrast(0.1)                                                                    # Randomly adjust contrast up to 10%
        ])

        return data_augmentation
    
    def create_cnn(self, optimiser="adam", batch_normalisation=False, learning_rate=0.001, decay_factor=0.95) -> tf.keras.Model:
        # Get data augmentation
        data_augmentation = self.get_augmentation()

        # Create a builder
        builder = CNNBuilder(input_shape=(32, 32, 3))
        
        if batch_normalisation:
            model = (builder
                    .add_data_augmentation(data_augmentation)
                    .add_conv_layer(64, (3,3))
                    .add_conv_layer(64, (3, 3))
                    .add_pooling_layer(strides=(2, 2))
                    .add_conv_layer(64, (3, 3))
                    .add_conv_layer(128, (3, 3))
                    .add_pooling_layer(strides=(2, 2))
                    .add_conv_layer(64, (3, 3))
                    .add_conv_layer(64, (3, 3))
                    .add_pooling_layer(strides=(2, 2))
                    .add_dropout(0.3)
                    .add_flaten_layer()
                    .add_dense_layer(256, activation="relu")
                    .add_dropout(0.3)
                    .add_dense_layer(10, activation="softmax")
                    .compile_model(optimiser=optimiser, learning_rate=learning_rate, decay_factor=decay_factor, use_lr_warmup=True, use_early_stopping=True)
                    .build()
                    )
        else:   
            # Create a base model
            model = (builder
                    .add_data_augmentation(data_augmentation)
                    .add_conv_layer(32, (3,3))
                    .add_pooling_layer()
                    .add_conv_layer(64, (3, 3))
                    .add_pooling_layer()
                    .add_flaten_layer()
                    .add_dense_layer(128, activation="relu")
                    .add_dense_layer(10, activation="softmax")
                    .compile_model(optimiser=optimiser, learning_rate=learning_rate, decay_factor=decay_factor)
                    .build()
                    )

        # Summarise the model
        model.summary()

        return model

    def compute_confusion_matrix(self, model, x_test, y_test):
        # Convert one-hot encoded labels to class indices
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Produce predictions
        y_pred = np.argmax(model.predict(x_test), axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred)

        return cm
    
    def get_diagonal_confusion_matrix(self, conf_matrix):
        # Get the diagonal values (true positives)
        diagonal = np.diag(conf_matrix)
        return diagonal
