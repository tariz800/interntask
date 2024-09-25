
# Image Segmentation and Object Identification Pipeline

This project provides a comprehensive pipeline for image segmentation, object extraction, identification, and attribute summarization using advanced machine learning techniques. The workflow involves uploading an image, segmenting it, extracting objects, identifying those objects with a YOLOv5 model, and summarizing the attributes of the identified objects.

## Table of Contents

- ### installation
- ### usage
- ### how-to-run-the-application
- ### contributing
- ### license

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-segmentation-object-identification.git
   cd image-segmentation-object-identification
Create a virtual environment (optional but recommended):

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. **Install required dependencies: Make sure you have pip installed. You can then run:**
   ```bash
   pip install -r requirements.txt

6. **Run streamlit file:**
   ```bash
   streamlit run app.py
    
## Usage

**Upload an Image:**

### Use the web interface to upload an image (in formats such as .jpg, .jpeg, or .png).**

![](https://github.com/tariz800/AI-Pipeline-for-Image-Segmentation-and-Object-Analysis/blob/main/assets/Screenshot%20(164).png)

**Image Segmentation:**
The application will automatically segment the uploaded image.

![](https://github.com/tariz800/AI-Pipeline-for-Image-Segmentation-and-Object-Analysis/blob/main/assets/Screenshot%20(165).png)

**Object Extraction:**
Extracted objects will be displayed and saved in your local directory along with their IDs.

![](https://github.com/tariz800/AI-Pipeline-for-Image-Segmentation-and-Object-Analysis/blob/main/assets/Screenshot%20(166).png)


**Object Identification:**
Each extracted object will be identified using the YOLOv5 model, with labels and confidence scores displayed. and its metadata will be stored in your local directory.

![](https://github.com/tariz800/AI-Pipeline-for-Image-Segmentation-and-Object-Analysis/blob/main/assets/Screenshot%20(167).png)

**Text Extraction:**
This will extract the text display and also saved the extracted text into your local directory in json format.

**Attribute Summarization:**
The application summarizes the attributes of identified objects, which can be viewed on the UI.

**Final Output:**
A final output image with annotations and a data table will be generated and displayed.

![](https://github.com/tariz800/AI-Pipeline-for-Image-Segmentation-and-Object-Analysis/blob/main/assets/Screenshot%20(169).png)
