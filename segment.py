from PIL import Image
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
import io

class ImageSegmenter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def segment_image(self, image_input):
        # Check if the input is a file path or an in-memory image
        if isinstance(image_input, str):  # It's a file path
            image = Image.open(image_input).convert("RGB")
        else:  # It's already a PIL image object
            image = image_input.convert("RGB")
    
        # Preprocess the image
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
    
        # Perform inference
        with torch.no_grad():
            prediction = self.model(image_tensor)
    
        # Extract masks, boxes, scores, and labels
        masks = prediction[0]['masks']  # Shape: (N, 1, H, W)
        boxes = prediction[0]['boxes'].cpu().numpy()  # Shape: (N, 4)
        scores = prediction[0]['scores'].cpu().numpy()  # Shape: (N,)
        labels = prediction[0]['labels'].cpu().numpy()  # Shape: (N,)
    
        # Filter out low-confidence predictions
        score_threshold = 0.7
        high_confidence_indices = scores > score_threshold
    
        masks = masks[high_confidence_indices]  # Still has shape (N, 1, H, W)
        boxes = boxes[high_confidence_indices]
        labels = labels[high_confidence_indices]
    
        # Remove the extra dimension for masks, but ensure we don't collapse into 2D
        masks = masks[:, 0, :, :].cpu().numpy()  # Now (N, H, W)
    
        return image, masks, boxes, labels


    def visualize_segmentation(self, image, masks, boxes, labels):
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(image))

        # Generate random colors for each instance
        colors = [tuple(np.random.rand(3)) for _ in range(len(masks))]

        for mask, box, label, color in zip(masks, boxes, labels, colors):
            # Create a boolean array from the mask
            bool_mask = mask > 0.5

            # Create a color mask
            color_mask = np.zeros((bool_mask.shape[0], bool_mask.shape[1], 4), dtype=np.uint8)
            color_mask[bool_mask] = [int(x * 255) for x in color] + [128]  # 128 for alpha

            # Overlay the color mask on the image
            plt.imshow(color_mask)

            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              fill=False, edgecolor=color, linewidth=2))

            # Add label
            plt.text(x1, y1, f"Class {label}", color='white', backgroundcolor=color,
                     fontsize=8, weight='bold')

        plt.axis('off')

        # Instead of saving, convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img

    def process_image(self, image_input):
        # Segment the image
        original_image, masks, boxes, labels = self.segment_image(image_input)

        # Visualize the segmentation
        segmented_image = self.visualize_segmentation(original_image, masks, boxes, labels)

        return original_image, segmented_image, masks, boxes, labels
