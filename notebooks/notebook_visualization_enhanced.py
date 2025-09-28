#!/usr/bin/env python3
"""
Enhanced visualization functions for the notebook that show rotated bounding boxes.
Based on the existing visualize_training_data.py but with rotated bbox support.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Any

def visualize_image_with_annotations_enhanced(image_path: str, annotations: List[Dict], 
                                            image_info: Dict, save_path: str = None) -> None:
    """Enhanced version that shows both axis-aligned and rotated bounding boxes."""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    #fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(15, 7))
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))
    
    # Show original image
    #ax1.imshow(image_rgb)
    #ax1.set_title(f"Original Image\\n{Path(image_path).name}")
    #ax1.axis('off')
    
    # Show image with annotations
    ax1.imshow(image_rgb)
    ax1.set_title(f"With Annotations ({len(annotations)} cards)")
    ax1.axis('off')
    
    # Colors for different cards
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(annotations), 1)))
    
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        
        # Draw axis-aligned bounding box (thin, semi-transparent)
        bbox = ann['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=1, edgecolor='red', facecolor='none', alpha=0.6
        )
        #ax2.add_patch(rect)
        
        # Draw rotated bounding box if available (thick, prominent)
        if 'rotated_bbox' in ann:
            rbbox = ann['rotated_bbox']  # [x1, y1, x2, y2, x3, y3, x4, y4]
            # Convert to polygon points
            points = [(rbbox[j], rbbox[j+1]) for j in range(0, len(rbbox), 2)]
            polygon = patches.Polygon(points, linewidth=2, edgecolor='green', 
                                    facecolor='none', alpha=0.9)
            ax1.add_patch(polygon)
            
            # Add rotation text
            if 'rotation' in ann:
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                ax1.text(center_x, center_y, f"{ann['rotation']:.1f}°", 
                       color='green', fontsize=8, weight='bold', ha='center')
        
        # Draw segmentation mask
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation'][0]  # First segmentation
            if len(segmentation) >= 6:  # At least 3 points (x,y pairs)
                # Convert to polygon points
                seg_points = np.array(segmentation).reshape(-1, 2)
                seg_polygon = patches.Polygon(
                    seg_points, closed=True, alpha=0.3, 
                    facecolor=color, edgecolor=color, linewidth=1
                )
                ax1.add_patch(seg_polygon)
        
        # Add annotation ID
        ax1.text(bbox[0], bbox[1] - 5, f'Card {i+1}', 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='red', label='Axis-Aligned BBox'),
        patches.Patch(facecolor='none', edgecolor='green', label='Rotated BBox')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_sample_images_notebook(data, images_dir, split_name, num_samples=6):
    """Notebook-friendly version that uses the enhanced visualization."""
    
    # Select random samples
    sample_indices = np.random.choice(len(data['images']), min(num_samples, len(data['images'])), replace=False)
    
    # Create image ID to annotations mapping
    image_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"\\n{split_name} Dataset Sample Visualizations")
    print("=" * 50)
    
    for i, idx in enumerate(sample_indices):
        img_info = data['images'][idx]
        img_path = images_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        annotations = image_annotations.get(img_info['id'], [])
        
        print(f"\\nSample {i+1}: {img_info['file_name']} ({len(annotations)} cards)")
        
        # Show enhanced visualization
        visualize_image_with_annotations_enhanced(
            str(img_path), annotations, img_info
        )


# Notebook usage code
NOTEBOOK_CODE = '''
# Import the enhanced visualization functions
import sys
sys.path.append('scripts')
from notebook_visualization_enhanced import visualize_sample_images_notebook

# Use the enhanced visualization function
print("Enhanced Sample Visualizations with Rotated Bounding Boxes")
print("=" * 60)

# Visualize samples from each split
visualize_sample_images_notebook(train_data, IMAGES_DIR, "Train", 6)
visualize_sample_images_notebook(val_data, IMAGES_DIR, "Validation", 3)
visualize_sample_images_notebook(test_data, IMAGES_DIR, "Test", 3)
'''

if __name__ == "__main__":
    print("Enhanced Notebook Visualization Functions")
    print("=" * 50)
    print("This module provides enhanced visualization functions for the notebook.")
    print("\\nTo use in your notebook, add this code to a new cell:")
    print()
    print(NOTEBOOK_CODE)
