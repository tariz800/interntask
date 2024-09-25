import os
import uuid
from PIL import Image
import json

class ObjectExtractor:
    def __init__(self, output_dir='extracted_objects', metadata_dir='metadata'):
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.ensure_dirs()

    # Ensure that the output directories exist
    def ensure_dirs(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)

    # Extract objects from image and save them
    def extract_objects(self, image, masks, boxes, labels):
        master_id = str(uuid.uuid4())  # Generate a unique ID for the master image
        extracted_objects = []

        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            object_id = str(uuid.uuid4())  # Generate a unique ID for each object
            
            # Extract object using the bounding box
            x1, y1, x2, y2 = map(int, box)
            object_image = image.crop((x1, y1, x2, y2))
            
            # Save the cropped object image
            file_name = f"{object_id}.png"
            file_path = os.path.join(self.output_dir, file_name)
            object_image.save(file_path)
            
            # Store metadata
            object_data = {
                'id': object_id,
                'master_id': master_id,
                'file_path': file_path,
                'bbox': box.tolist(),
                'label': int(label)
            }
            extracted_objects.append(object_data)
            
            # Save metadata as a JSON file
            self.save_metadata(object_data)

        return master_id, extracted_objects

    # Save object metadata to the local file system as a JSON file
    def save_metadata(self, object_data):
        metadata_file = os.path.join(self.metadata_dir, f"{object_data['id']}.json")
        with open(metadata_file, 'w') as json_file:
            json.dump(object_data, json_file, indent=4)

    # Retrieve object metadata from the local file system using object ID
    def get_object_by_id(self, object_id):
        metadata_file = os.path.join(self.metadata_dir, f"{object_id}.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as json_file:
                object_data = json.load(json_file)
            return object_data
        return None

# Example usage:
# extractor = ObjectExtractor()
# master_id, extracted_objects = extractor.extract_objects(image, masks, boxes, labels)
# for obj in extracted_objects:
#     print(f"Extracted object: {obj['id']}, from master image: {obj['master_id']}")
#
# # Retrieve an object
# retrieved_object = extractor.get_object_by_id(extracted_objects[0]['id'])
# print(f"Retrieved object: {retrieved_object}")
