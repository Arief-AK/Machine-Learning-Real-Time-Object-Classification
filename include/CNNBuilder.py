import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, regularizers

from include.Logger import Logger

class CNNBuilder:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.logger = Logger(__name__)

    def add_conv_layer(self, filters, kernel_size, activation="relu", padding="same", kernel_reguliser=None):
        self.model.add(layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_regularizer=kernel_reguliser))
        return self
    
    def add_pooling_layer(self, pool_size=(2, 2), strides=None, pool_type="max"):
        if pool_type == "max":
            self.model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))
        elif pool_type == "avg":
            self.model.add(layers.AveragePooling2D(pool_size=pool_size, strides=strides))
        else:
            self.logger.error("Unrecognised pool type")
            exit(-1)

        return self
    
    def add_batch_norm(self):
        self.model.add(layers.BatchNormalization())
        return self

    def add_dropout(self, rate):
        self.model.add(layers.Dropout(rate))
        return self

    def add_flaten_layer(self):
        self.model.add(layers.Flatten())
        return self

    def add_dense_layer(self, units, activation):
        self.model.add(layers.Dense(units, activation=activation))
        return self
    
    def add_data_augmentation(self, augmentation):
        self.model.add(augmentation)
        return self

    def compile_model(self, optimiser="adam", loss="categorical_crossentropy", metrics=['accuracy']):
        self.model.compile(optimizer=optimiser, loss=loss, metrics=metrics)
        return self

    def build(self):
        return self.model