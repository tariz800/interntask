import os
import json
from object_identification import ObjectIdentifier

# Initialize the object identifier
identifier = ObjectIdentifier()

# Path to the folder containing extracted object images
extracted_objects_dir = 'extracted_objects'
output_file = 'identified_objects.json'

# Automatically gather all image files in the extracted_objects folder
extracted_objects = []
for file_name in os.listdir(extracted_objects_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for common image formats
        object_data = {
            'id': file_name.split('.')[0],  # Use file name (without extension) as object ID
            'file_path': os.path.join(extracted_objects_dir, file_name)  # Full path to the image file
        }
        extracted_objects.append(object_data)

# Identify objects in the extracted images
identified_objects = identifier.identify_objects(extracted_objects)

# Print the identified objects and their labels
for obj in identified_objects:
    print(f"Object ID: {obj['id']}")
    print(f"Labels: {obj['labels']}")
    print(f"Confidence Scores: {obj['confidences']}")
    print(f"File Path: {obj['file_path']}")
    print("----")

# Save the identified objects to a JSON file
with open(output_file, 'w') as f:
    json.dump(identified_objects, f, indent=4)

print(f"Identified objects have been saved to {output_file}.")


import json

with open('identified_objects.json', 'r') as f:
    identified_objects = json.load(f)

# Now `identified_objects` is a list of dictionaries with the identified data for each object
for obj in identified_objects:
    print(f"Processing object {obj['id']} with labels: {obj['labels']}")
