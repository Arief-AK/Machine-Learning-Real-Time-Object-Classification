import tensorflow as tf
import numpy as np
import time

from include.Logger import Logger, logging

class ModelProfiler:
    def __init__(self):
        self.logger = Logger(__name__)
        self.logger.set_level(logging.INFO)

    def _profile_average_inference_time(self, model, x_sample, num_runs=100):
        # Initialise time list
        times = []

        # Iterate through the number of runs
        for _ in range(num_runs):
            # Measure the inference time
            start_time = time.perf_counter()
            _ = model.predict(x_sample, verbose=0)
            end_time = time.perf_counter()

            # Append the time to the list
            times.append((end_time - start_time) * 1000)

        avg_time = np.mean(times)
        return avg_time

    def measure_average_inference_time(self, batch_size, model, x_sample, show_single_image_inference=False) -> tuple:
        # Show the inference time for a single image
        if show_single_image_inference:
            x_single = np.expand_dims(x_sample[0], axis=0)
            single_image_time = self._profile_average_inference_time(model, x_single)
            self.logger.debug(f"Single image inference time: {single_image_time:.2f}ms")
        
        # Set the batch size
        x_batch = x_sample[:batch_size]
        batch_time = self._profile_average_inference_time(model, x_batch)
        self.logger.debug(f"Batch inference time: {batch_time:.2f}ms")

        # Calculate the throughput
        throughput = batch_size / (batch_time / 1000)
        self.logger.debug(f"Throughput: {throughput:.2f} images/s")
        return batch_time, throughput