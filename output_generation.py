import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from PIL import Image

class OutputGenerator:
    def __init__(self, original_image_path, mapped_data_file='mapped_data.json', output_image_file='final_output.png'):
        self.original_image_path = original_image_path
        self.mapped_data_file = mapped_data_file
        self.output_image_file = output_image_file

    def load_mapped_data(self):
        # Load the mapped data from the JSON file
        with open(self.mapped_data_file, 'r') as f:
            self.mapped_data = json.load(f)

    def annotate_image(self):
        # Open the original image
        original_image = Image.open(self.original_image_path)
        fig, ax = plt.subplots(1, figsize=(10, 10),dpi=600)
        ax.imshow(original_image)

        # Add bounding boxes and labels from the mapped data
        for obj in self.mapped_data:
            bbox = obj['bbox']
            label = obj['labels'][0]  # Assuming one label per object

            # Create a rectangle patch for each bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),  # x, y
                bbox[2] - bbox[0],  # width
                bbox[3] - bbox[1],  # height
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Add the label as text above the bounding box
            ax.text(
                bbox[0], bbox[1] - 10,
                label,
                color='white',
                fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5)
            )

        plt.axis('off')
        plt.savefig(self.output_image_file, bbox_inches='tight', pad_inches=0)
        plt.close()

    def generate_table(self, table_file='final_table.png'):
        # Prepare data for the table
        table_data = []
        for obj in self.mapped_data:
            table_data.append({
                'ID': obj['id'],
                'Label': obj['labels'][0],
                'Confidence': round(obj['confidences'][0], 2),
                'Extracted Text': obj.get('extracted_text', 'N/A'),
                'Summary': obj.get('summary', 'N/A')  # Ensure this contains only the summarized text
            })
    
        # Create a DataFrame for the table
        df = pd.DataFrame(table_data)
    
        # Plot the table using matplotlib
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.6), dpi=600)  # Adjust height based on the number of rows
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
        # Save the table as an image
        plt.savefig(table_file, bbox_inches='tight', pad_inches=0.5)
        plt.close()


    def generate_final_output(self, final_output_file='final_output_with_table.png'):
        # Generate the annotated image and the table image
        self.annotate_image()
        self.generate_table()

        # Combine the annotated image and table image side by side
        annotated_image = Image.open(self.output_image_file)
        table_image = Image.open('final_table.png')

        # Create a new image large enough to hold both the annotated image and the table
        total_width = annotated_image.width + table_image.width
        max_height = max(annotated_image.height, table_image.height)
        combined_image = Image.new('RGB', (total_width, max_height))

        # Paste the images side by side
        combined_image.paste(annotated_image, (0, 0))
        combined_image.paste(table_image, (annotated_image.width, 0))

        # Save the combined final output
        combined_image.save(final_output_file)
        print(f"Final output saved as {final_output_file}.")

# Example usage
if __name__ == "__main__":
    # Path to the original image
    original_image_path = "E:/Project/warsoft_task/Aviation-careers.jpg"

    # Initialize and run the output generator
    output_generator = OutputGenerator(original_image_path)
    output_generator.load_mapped_data()
    output_generator.generate_final_output('final_output_with_table.png')