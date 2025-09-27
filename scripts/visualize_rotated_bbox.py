#!/usr/bin/env python3
"""
Visualization script to show rotated bounding boxes on generated training images.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def draw_rotated_bbox(image, rotated_bbox, color=(0, 255, 0), thickness=2):
    """Draw a rotated bounding box on an image."""
    # Convert flat list to points
    points = []
    for i in range(0, len(rotated_bbox), 2):
        points.append([int(rotated_bbox[i]), int(rotated_bbox[i+1])])
    
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], True, color, thickness)
    return image

def draw_axis_aligned_bbox(image, bbox, color=(255, 0, 0), thickness=2):
    """Draw an axis-aligned bounding box on an image."""
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return image

def visualize_annotations(data_dir, output_dir, split='train', max_images=5):
    """Visualize annotations on images."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load annotations
    annotations_file = data_path / "annotations" / f"{split}_annotations.json"
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image ID to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process images
    processed = 0
    for image_info in coco_data['images']:
        if processed >= max_images:
            break
            
        image_id = image_info['id']
        filename = image_info['file_name']
        
        # Load image
        image_path = data_path / "images" / filename
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Create visualization copies
        vis_rotated = image.copy()
        vis_comparison = image.copy()
        
        # Draw annotations if they exist
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                # Draw rotated bbox (green)
                if 'rotated_bbox' in ann:
                    draw_rotated_bbox(vis_rotated, ann['rotated_bbox'], (0, 255, 0), 2)
                    draw_rotated_bbox(vis_comparison, ann['rotated_bbox'], (0, 255, 0), 2)
                
                # Draw axis-aligned bbox (red) for comparison
                draw_axis_aligned_bbox(vis_comparison, ann['bbox'], (0, 0, 255), 2)
                
                # Add rotation text
                if 'rotation' in ann:
                    rotation = ann['rotation']
                    bbox = ann['bbox']
                    text_x = int(bbox[0])
                    text_y = int(bbox[1] - 10)
                    cv2.putText(vis_comparison, f"Rot: {rotation:.1f}°", 
                              (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 255), 1)
        
        # Save visualizations
        base_name = Path(filename).stem
        
        # Save rotated bbox only
        rotated_output = output_path / f"{base_name}_rotated.jpg"
        cv2.imwrite(str(rotated_output), vis_rotated)
        
        # Save comparison (both bbox types)
        comparison_output = output_path / f"{base_name}_comparison.jpg"
        cv2.imwrite(str(comparison_output), vis_comparison)
        
        print(f"Processed {filename} -> {base_name}_rotated.jpg, {base_name}_comparison.jpg")
        processed += 1
    
    print(f"\nVisualization complete! Generated {processed * 2} images in {output_dir}")
    print("Files:")
    print("  *_rotated.jpg: Shows only rotated bounding boxes (green)")
    print("  *_comparison.jpg: Shows both rotated (green) and axis-aligned (red) bboxes")

def main():
    parser = argparse.ArgumentParser(description="Visualize rotated bounding boxes")
    parser.add_argument("--data_dir", type=str, default="data/review_batch", 
                       help="Directory containing images and annotations")
    parser.add_argument("--output_dir", type=str, default="data/review_batch/visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                       help="Dataset split to visualize")
    parser.add_argument("--max_images", type=int, default=5,
                       help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    visualize_annotations(args.data_dir, args.output_dir, args.split, args.max_images)

if __name__ == "__main__":
    main()
