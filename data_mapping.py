import json
import os

class DataMapper:
    def __init__(self, identified_objects_file='identified_objects.json', summarized_attributes_file='summarized_attributes.json', metadata_dir='metadata'):
        self.identified_objects_file = identified_objects_file
        self.summarized_attributes_file = summarized_attributes_file
        self.metadata_dir = metadata_dir

    def load_data(self):
        # Load identified objects
        with open(self.identified_objects_file, 'r') as f:
            self.identified_objects = json.load(f)

        # Load summarized attributes
        with open(self.summarized_attributes_file, 'r') as f:
            self.summarized_attributes = json.load(f)

    def load_metadata(self, object_id):
        # Load metadata from the corresponding file in the metadata folder
        metadata_file = os.path.join(self.metadata_dir, f"{object_id}.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            print(f"Metadata file for {object_id} not found.")
            return None

    def map_data(self):
        mapped_data = []

        for obj in self.identified_objects:
            object_id = obj['id']
            
            # Load metadata to get the master_id and bounding box
            metadata = self.load_metadata(object_id)
            if metadata is None:
                continue  # Skip if metadata is missing
            
            master_id = metadata['master_id']

            # Find the corresponding summary for this object by ID
            summary = next((item for item in self.summarized_attributes if item['id'] == object_id), None)
            if not summary:
                print(f"Summary for object {object_id} not found.")
                continue

            # Combine all relevant data
            mapped_object = {
                "id": object_id,
                "master_id": master_id,
                "file_path": obj['file_path'],
                "labels": obj['labels'],
                "confidences": obj['confidences'],
                "bbox": metadata['bbox'],
                "extracted_text": summary['extracted_text'],
                "summary": summary['summary']
            }

            mapped_data.append(mapped_object)

        return mapped_data

    def save_mapped_data(self, output_file='mapped_data.json'):
        mapped_data = self.map_data()
        with open(output_file, 'w') as f:
            json.dump(mapped_data, f, indent=4)
        print(f"Mapped data saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    data_mapper = DataMapper()
    data_mapper.load_data()
    data_mapper.save_mapped_data('mapped_data.json')
