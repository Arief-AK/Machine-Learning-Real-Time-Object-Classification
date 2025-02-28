import cv2
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.models import load_model # type: ignore
from include.Logger import Logger

class LiveCameraClassifier:
    def __init__(self, model, class_names, logger=None):
        self.model = model
        self.class_names = class_names
        self.logger = logger if logger else Logger(__name__)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Set width
        self.cap.set(4, 480)  # Set height

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (32, 32))
        frame = frame.astype("float32") / 255.0
        return np.expand_dims(frame, axis=0)

    def draw_overlay(self, frame, label, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (150, 100), (490, 380), (0, 255, 0), 2)  # Bounding box
        return frame

    def run(self):
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                break

            processed_frame = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_frame, verbose=0)
            label = self.class_names[np.argmax(predictions)]

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            frame = self.draw_overlay(frame, label, fps)
            cv2.imshow("Live Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    model = load_model("models/batch_norm_model_rmsprop.keras")
    classifier = LiveCameraClassifier(model, class_names)
    classifier.run()