import os
import json
import easyocr
import re
class TextExtractor:
    def __init__(self, ocr_tool='easyocr'):
        if ocr_tool == 'easyocr':
            self.reader = easyocr.Reader(['en'])  # Load English OCR model
        else:
            raise ValueError("Currently only EasyOCR is supported.")

    def extract_text(self, image_path):
        # Perform OCR on the given image
        result = self.reader.readtext(image_path)
        extracted_text = ' '.join([res[1] for res in result])  # Extract the text
        return self.clean_text(extracted_text)  # Clean the extracted text

    def clean_text(self, text):
        # Remove unwanted characters (e.g., brackets, special characters)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric characters and spaces
        return cleaned_text.strip()  # Remove leading/trailing whitespace

    def extract_from_objects(self, identified_objects):
        extracted_data = []

        for obj in identified_objects:
            obj_id = obj['id']
            file_path = obj['file_path']
            
            if os.path.exists(file_path):
                print(f"Extracting text from object {obj_id} at {file_path}...")
                
                # Extract text from the object image
                text = self.extract_text(file_path)
                
                extracted_data.append({
                    'id': obj_id,
                    'file_path': file_path,
                    'extracted_text': text
                })
            else:
                print(f"Error: File {file_path} does not exist.")
        
        return extracted_data

    def save_extracted_text(self, extracted_data, output_file='extracted_text.json'):
        # Save extracted text to a JSON file
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        print(f"Extracted text has been saved to {output_file}.")
