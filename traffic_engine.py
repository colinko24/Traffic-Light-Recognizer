import cv2
import numpy as np
from tensorflow import keras
import object_detection

class TrafficSystem:
    def __init__(self, classifier_path="traffic.keras"):
        """Initialize models once to save memory and startup time."""
        # Load the SSD ResNet50 model trained on COCO
        self.detector = object_detection.load_ssd_coco()
        # Load your custom color classifier
        self.classifier = keras.models.load_model(classifier_path)
        print("Traffic Light Recognition System Initialized.")

    def process_frame(self, frame):
        """Standardized processing for a single image or video frame."""
        # This calls the helper function that performs both detection and classification
        return object_detection.perform_object_detection_video(
            self.detector, frame, model_traffic_lights=self.classifier
        )

    def extract_and_save(self, frame, counter, output_folder="traffic_light_cropped"):
        """Consolidates logic from extract_traffic_lights.py for dataset building."""
        # Uses the detector without the classifier to find raw lights
        _, out, _ = object_detection.perform_object_detection(
            self.detector, "temp_img", save_annotated=False
        )
        
        saved_count = 0
        for idx, box in enumerate(out['boxes']):
            if out["detection_classes"][idx] == object_detection.LABEL_TRAFFIC_LIGHT:
                # Crop logic: [y:y2, x:x2]
                crop = frame[box["y"]:box["y2"], box["x"]:box["x2"]]
                cv2.imwrite(f"{output_folder}/{counter}_{idx}.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                saved_count += 1
        return saved_count
