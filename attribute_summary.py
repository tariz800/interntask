import json

class AttributeSummarizer:
    def __init__(self, extracted_text_file='extracted_text.json', identified_objects_file='identified_objects.json'):
        self.extracted_text_file = extracted_text_file
        self.identified_objects_file = identified_objects_file

    def load_data(self):
        with open(self.extracted_text_file, 'r') as f:
            self.extracted_data = json.load(f)

        with open(self.identified_objects_file, 'r') as f:
            self.identified_objects = json.load(f)

    def summarize_attributes(self):
        summary = []

        # Create a mapping of object ID to extracted text
        text_mapping = {item['id']: item['extracted_text'] for item in self.extracted_data}

        for obj in self.identified_objects:
            obj_id = obj['id']
            file_path = obj['file_path']
            extracted_text = text_mapping.get(obj_id, "No text extracted")

            # Ensure label is a single value
            label = obj['labels']
            if isinstance(label, list):
                label = label[0]  # Take the first label if it's a list

            # Generate a detailed summary
            summary_entry = {
                'id': obj_id,
                'file_path': file_path,
                'label': label,
                'extracted_text': extracted_text,
                'summary': self.generate_summary(extracted_text, label)
            }
            summary.append(summary_entry)

        return summary

    def generate_summary(self, extracted_text, label):
        # Mapping of labels to descriptive terms
        label_descriptions = {
            1: "Airplane",
            2: "Discount Announcement",
            3: "Flight Details",
            # Add more mappings as needed
        }

        # Construct the summary
        description = label_descriptions.get(label, "Unknown Object")
        key_attributes = self.extract_key_attributes(extracted_text)

        summary = (
            f"Object ID: {description}\n"
            f"Description: This object is identified as a {description}.\n"
            f"Key Attributes: {key_attributes}\n"
            f"Extracted Text: {extracted_text if extracted_text != 'No text extracted' else 'N/A'}"
        )
        
        return summary

    def extract_key_attributes(self, extracted_text):
        # Extract meaningful attributes from the text
        # This is a placeholder; implement your own logic as needed
        if "airplane" in extracted_text.lower():
            return "Type: Commercial, Features: Advanced Technology"
        elif "discount" in extracted_text.lower():
            return "Type: Promotion, Features: Limited Time Offer"
        elif "flight" in extracted_text.lower():
            return "Type: Flight Information, Features: Departure and Arrival Details"
        else:
            return "General Information"

    def save_summary(self, summary, output_file='summarized_attributes.json'):
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Summarized attributes saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    summarizer = AttributeSummarizer()
    summarizer.load_data()
    summarized_attributes = summarizer.summarize_attributes()
    summarizer.save_summary(summarized_attributes)
