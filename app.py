import os
import streamlit as st
from PIL import Image
from segment import ImageSegmenter  # Import your segmentation module
from object_extractor import ObjectExtractor  # Import the extraction module
from object_identifier import ObjectIdentifier  # Import object identification module
from text_extractor import TextExtractor  # Import the text extractor module
from attribute_summary import AttributeSummarizer  # Import the attribute summarizer module
from data_mapping import DataMapper  # Import the data mapping module
from output_generation import OutputGenerator  # Import the output generation module
import json
import time

# Create the base output directory
base_output_dir = 'output'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)
def main():
    # Helper function to create subdirectories for each functionality
    def create_subdirectory(function_name):
        subdir = os.path.join(base_output_dir, function_name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        return subdir

    # Initialize the modules
    segmenter = ImageSegmenter()
    extractor = ObjectExtractor(output_dir='extracted_objects', metadata_dir='metadata')
    identifier = ObjectIdentifier()
    text_extractor = TextExtractor()
    summarizer = AttributeSummarizer()
    data_mapper = DataMapper()

    # Streamlit app UI with step headings
    st.title("Image Segmentation, Extraction, and Identification Pipeline")

    # Step 1: Upload Image
    st.header("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load the image and display it
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Step 2: Segment the image
        st.header("Step 2: Segment the Image")
        st.write("Segmenting the image...")

        segmentation_dir = create_subdirectory('segmentation')
        original_image, segmented_image, masks, boxes, labels = segmenter.process_image(image)

        # Save the segmented image
        segmented_image_path = os.path.join(segmentation_dir, 'segmented_image.png')
        segmented_image.save(segmented_image_path)
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        # Step 3: Extract objects from the image
        st.header("Step 3: Extract Objects from the Image")
        st.write("Extracting objects from the segmented image...")

        #extraction_dir = create_subdirectory('extraction')
        master_id, extracted_objects = extractor.extract_objects(original_image, masks, boxes, labels)

        # Display extracted objects
        st.write(f"Number of objects extracted: {len(extracted_objects)}")
        for obj in extracted_objects:
            st.image(obj['file_path'], caption=f"Object ID: {obj['id']}")
        
        # Step 4: Identify Objects
        st.header("Step 4: Identify Objects")
        st.write("Identifying objects using YOLOv5 model...")

        identification_dir = create_subdirectory('identification')
        identified_objects = identifier.identify_objects(extracted_objects)
        
        identified_objects_file = os.path.join(identification_dir, 'identified_objects.json')
        with open(identified_objects_file, 'w') as f:
            json.dump(identified_objects, f, indent=4)
        st.write(f"Identified objects have been saved to {identified_objects_file}.")

        # Display identified objects with confidence > 0.7
        filtered_objects = [obj for obj in identified_objects if max(obj['confidences']) > 0.7]
        if filtered_objects:
            for obj in filtered_objects:
                st.write(f"**Object ID**: {obj['id']}")
                st.write(f"**Labels**: {', '.join(obj['labels'])}")
                st.write(f"**Confidence Scores**: {', '.join([str(c) for c in obj['confidences']])}")
                st.image(obj['file_path'], caption=f"Object ID: {obj['id']} with Labels: {', '.join(obj['labels'])}")
        else:
            st.write("No identified objects with confidence greater than 0.7.")

        # Step 5: Extract text from the identified objects
        st.header("Step 5: Extract Text from Objects")
        st.write("Extracting text from the identified objects...")

        text_extraction_dir = create_subdirectory('text_extraction')
        extracted_text_data = text_extractor.extract_from_objects(identified_objects)
        extracted_text_file = os.path.join(text_extraction_dir, 'extracted_text.json')
        text_extractor.save_extracted_text(extracted_text_data, extracted_text_file)
        st.write(f"Extracted text has been saved to {extracted_text_file}.")
        
        # Step 6: Summarize attributes of objects
        st.header("Step 6: Summarize Object Attributes")
        st.write("Summarizing object attributes...")

        attribute_summary_dir = create_subdirectory('attribute_summary')
        summarizer.load_data()
        summarized_attributes = summarizer.summarize_attributes()

        summarized_attributes_file = os.path.join(attribute_summary_dir, 'summarized_attributes.json')
        summarizer.save_summary(summarized_attributes, output_file=summarized_attributes_file)
        st.write(f"Summarized attributes have been saved to {summarized_attributes_file}.")

        # Step 7: Map all extracted data to each object and the master input image
        st.header("Step 7: Map Data to Each Object")
        st.write("Mapping data to each object...")

        data_mapping_dir = create_subdirectory('data_mapping')
        data_mapper.load_data()
        mapped_output_file = os.path.join(data_mapping_dir, 'mapped_data.json')
        data_mapper.save_mapped_data(output_file=mapped_output_file)
        st.write(f"Mapped data has been save {mapped_output_file}")

                 # Step 8: Generate Final Output (Image with Annotations and Data Table)
        st.header("Step 8: Generate Final Output")
        st.write("Generating the final output with annotations and data table...")
        
        output_generation_dir = create_subdirectory('output_generation')
        
        # Path to the original image (in the correct format for the output generation)
        original_image_path = uploaded_file
        
        # Initialize and run the output generator
        output_generator = OutputGenerator(original_image_path)
        output_generator.load_mapped_data()
        
        final_output_file = os.path.join(output_generation_dir, 'final_output_with_table.png')
        output_generator.generate_final_output(final_output_file)
        
        # Display the final output
        st.image(final_output_file, caption="Final Output: Annotated Image with Data Table", use_column_width=True)
        st.write(f"Final output has been saved to {final_output_file}.")
    
if __name__ == '__main__':
    main()
