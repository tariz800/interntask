import os
import torch
from PIL import Image

class ObjectIdentifier:
    def __init__(self, model_name='yolov5s'):
        # Load YOLOv5 model (small version)
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    def identify_objects(self, extracted_objects):
        identified_objects = []

        # Loop through each extracted object image
        for obj in extracted_objects:
            image_path = obj['file_path']
            if not os.path.exists(image_path):
                print(f"Error: File {image_path} does not exist.")
                continue

            # Load the object image
            img = Image.open(image_path)
            
            # Perform object detection
            results = self.model(img)
            
            # Get the class label and confidence score
            predictions = results.pred[0]
            if len(predictions) > 0:
                labels = predictions[:, -1].int().tolist()  # Class indices
                confidences = predictions[:, 4].tolist()    # Confidence scores
                class_labels = [self.model.names[label] for label in labels]

                # Store the identification details for the object
                obj_description = {
                    'id': obj['id'],
                    'file_path': image_path,
                    'labels': class_labels,
                    'confidences': confidences
                }
                identified_objects.append(obj_description)
            else:
                print(f"No objects identified in {image_path}")

        return identified_objects


