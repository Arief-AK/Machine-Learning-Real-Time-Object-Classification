import os
import tensorflow as tf
import matplotlib.pyplot as plt

from include.Logger import Logger

class Visualiser:
    def __init__(self, image_directory="images/"):
        self.IMAGE_DIRECTORY = image_directory
        self.logger = Logger(__name__)

    def visualise_sample_images(self, num_samples: int, x_train, num_augmentations: int, data_augmentation: tf.keras.Sequential):
        # Normalise data
        x_train = x_train.astype("float32")  # Ensure float32
        x_train = (x_train * 255).astype("uint8")  # Convert to uint8 if needed

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}"
        os.makedirs(result_directory, exist_ok=True)

        plt.figure(figsize=(num_samples * 2, num_augmentations * 2))

        for i in range(num_samples):
            for j in range(num_augmentations):
                ax = plt.subplot(num_augmentations, num_samples, j * num_samples + i + 1)
                augmented_image = data_augmentation(tf.expand_dims(x_train[i], axis=0), training=True)

                augmented_image = tf.clip_by_value(augmented_image[0], 0, 255)
                augmented_image = augmented_image.numpy().astype("uint8")

                plt.imshow(augmented_image)
                plt.axis("off")

        plt.savefig(f"{result_directory}augmented_sample_image.png")