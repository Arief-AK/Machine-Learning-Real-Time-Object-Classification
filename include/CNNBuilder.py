from include.Logger import Logger
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers # type: ignore

class CustomEarlyStopping(callbacks.Callback):
    def __init__(self, target_accuracy=0.95, patience=5, restore_best_weights=True):
        super(CustomEarlyStopping, self).__init__()
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_accuracy = 0
        self.wait = 0
        self.logger = Logger(__name__)

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs["val_accuracy"]
        
        # Check the current accuracy against the best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait = 0
        else:
            self.wait += 1
            self.logger.debug(f"Accuracy: {current_accuracy}, Best Accuracy: {self.best_accuracy}, Wait: {self.wait}")
            
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
                    self.logger.info("Restored best weights")

class CNNBuilder:
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.logger = Logger(__name__)

    def add_conv_layer(self, filters, kernel_size, activation="relu", padding="same", kernel_reguliser=None, weight_decay=1e-4):
        kernel_reguliser = kernel_reguliser if kernel_reguliser else None
        self.model.add(layers.Conv2D(filters, kernel_size, strides=1, padding=padding, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer="he_normal"))  # He initialization
        self.model.add(layers.BatchNormalization())     # BatchNorm before activation
        self.model.add(layers.Activation(activation))   # Separate activation
        return self

    def add_pooling_layer(self, pool_size=(2, 2), strides=None, pool_type="max"):
        if pool_type == "max":
            self.model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides))
        elif pool_type == "avg":
            self.model.add(layers.AveragePooling2D(pool_size=pool_size, strides=strides))
        else:
            self.logger.error("Unrecognized pool type")
            exit(-1)
        return self

    def add_dropout(self, rate):
        self.model.add(layers.Dropout(rate))
        return self

    def add_flaten_layer(self):
        self.model.add(layers.Flatten())
        return self

    def add_dense_layer(self, units, activation='relu'):
        self.model.add(layers.Dense(units, kernel_initializer="he_normal"))     # He initialization
        self.model.add(layers.BatchNormalization())                             # BatchNorm before activation
        self.model.add(layers.Activation(activation))                           # Separate activation
        return self

    def add_data_augmentation(self, augmentation):
        self.model.add(augmentation)
        return self

    def compile_model(self, optimiser="adam", learning_rate=0.001, decay_factor=0.95, loss="categorical_crossentropy", metrics=['accuracy'], use_lr_warmup=False, use_early_stopping=False):
        if isinstance(optimiser, str):
            if optimiser == "adam":
                optimiser = optimizers.AdamW(learning_rate=learning_rate)               # Switched to AdamW
            elif optimiser == "sgd":
                optimiser = optimizers.SGD(learning_rate=learning_rate, momentum=0.95)  # Increased momentum
            elif optimiser == "rmsprop":
                optimiser = optimizers.RMSprop(learning_rate=learning_rate, rho=0.98)   # Adjusted decay factor
            else:
                self.logger.error("Failed to compile model")
                exit(-1)

        # Compile the model
        self.model.compile(optimizer=optimiser, loss=loss, metrics=metrics)

        # Adaptive learning rate scheduler based on optimizer type
        if isinstance(optimiser, (optimizers.AdamW, optimizers.RMSprop)):
            decay_factor = 0.98                                                     # AdamW & RMSprop benefit from a slightly higher decay
        elif isinstance(optimiser, optimizers.SGD):
            decay_factor = 0.90                                                     # Faster decay for SGD to adapt quickly

        self.lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: learning_rate * (decay_factor ** epoch))

        # Learning Rate Warm-up
        self.lr_warmup = None
        if use_lr_warmup:
            def lr_schedule(epoch, lr):
                if epoch < 10:
                    return 0.001
                elif epoch < 30:
                    return 0.0005
                else:
                    return 0.0002
            self.lr_warmup = callbacks.LearningRateScheduler(lr_schedule)

        # Early Stopping
        self.early_stopping = None
        if use_early_stopping:
            self.early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        # Get all callbacks
        self.callbacks_list = self.get_callbacks()
        return self

    def get_callbacks(self):
        callbacks_list = [self.lr_scheduler]
        if self.lr_warmup:
            callbacks_list.append(self.lr_warmup)
        if self.early_stopping:
            callbacks_list.append(self.early_stopping)
        return callbacks_list

    def build(self):
        return self.model
