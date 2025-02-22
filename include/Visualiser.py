import os
import seaborn as sns
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

        plt.figure(figsize=(num_samples * 2, num_augmentations * 2))

        for i in range(num_samples):
            for j in range(num_augmentations):
                ax = plt.subplot(num_augmentations, num_samples, j * num_samples + i + 1)
                augmented_image = data_augmentation(tf.expand_dims(x_train[i], axis=0), training=True)

                augmented_image = tf.clip_by_value(augmented_image[0], 0, 255)
                augmented_image = augmented_image.numpy().astype("uint8")

                plt.imshow(augmented_image)
                plt.axis("off")

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}"
        os.makedirs(result_directory, exist_ok=True)
        plt.savefig(f"{result_directory}augmented_sample_image.png")

    def plot_training_history(self, history, str_model):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax[0].plot(history.history["accuracy"], label="Train Accuracy")
        ax[0].plot(history.history["val_accuracy"], label="Val Accuracy")
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        # Loss plot
        ax[1].plot(history.history["loss"], label="Train Loss")
        ax[1].plot(history.history["val_loss"], label="Val Loss")
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}Training/{str_model}/"
        os.makedirs(result_directory, exist_ok=True)
        plt.savefig(f"{result_directory}{str_model}_training_history.png")
    
    def plot_confusion_matrix(self, matrix, class_names: list, str_model):
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}Training/{str_model}/"
        os.makedirs(result_directory, exist_ok=True)
        plt.savefig(f"{result_directory}{str_model}_training_confusion_matrix.png")

    def plot_diagonal_confusion_matrix(self, diagonal, class_names: list, str_model):
        # Create a plot
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(diagonal)), diagonal, color='skyblue', edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('True Positives')
        plt.title('Confusion Matrix Diagonal (True Positives)')
        
        # Set lables for x-axis to class names
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        
        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}Training/{str_model}/"
        os.makedirs(result_directory, exist_ok=True)
        plt.savefig(f"{result_directory}{str_model}_training_diagonal_confusion_matrix.png")

    def plot_inference_timings(self, inference_timings: list, batch_size:int, str_model):
        metrics = ["Single Image Inference Time", f"Batch Inference Time ({batch_size})", "Throughput (Images/Sec)"]
        
        plt.figure(figsize=(8, 5))
        plt.bar(metrics, inference_timings, color=['blue', 'green', 'red'])
        plt.ylabel("Milliseconds (ms) / Images per sec")
        plt.title("Inference Speed Comparison")
        plt.xticks(rotation=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Display the values on top of the bars
        for i, v in enumerate(inference_timings):
            plt.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=12)

        # Create result directory
        result_directory = f"{self.IMAGE_DIRECTORY}Inference/{str_model}/"
        os.makedirs(result_directory, exist_ok=True)
        plt.savefig(f"{result_directory}{str_model}_{batch_size}_inference_timings.png")