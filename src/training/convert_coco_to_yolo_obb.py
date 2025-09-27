#!/usr/bin/env python3
"""
Convert COCO format annotations with rotated bounding boxes to YOLO OBB format.
YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def normalize_coordinates(coords: List[float], img_width: int, img_height: int) -> List[float]:
    """Normalize coordinates to [0, 1] range."""
    normalized = []
    for i in range(0, len(coords), 2):
        x = coords[i] / img_width
        y = coords[i + 1] / img_height
        normalized.extend([x, y])
    return normalized


def convert_rotated_bbox_to_yolo_obb(rotated_bbox: List[float], img_width: int, img_height: int) -> str:
    """
    Convert rotated bbox (8 coordinates) to YOLO OBB format.
    
    Args:
        rotated_bbox: [x1, y1, x2, y2, x3, y3, x4, y4] - 4 corner points
        img_width: Image width
        img_height: Image height
    
    Returns:
        YOLO OBB format string: "class_id x1 y1 x2 y2 x3 y3 x4 y4"
    """
    # Normalize coordinates
    normalized_coords = normalize_coordinates(rotated_bbox, img_width, img_height)
    
    # Class ID is 0 for "card" (single class)
    class_id = 0
    
    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    coords_str = " ".join([f"{coord:.6f}" for coord in normalized_coords])
    return f"{class_id} {coords_str}"


def convert_coco_to_yolo_obb(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    split_name: str
) -> None:
    """
    Convert COCO annotations to YOLO OBB format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO labels
        split_name: Split name (train/val/test)
    """
    print(f"Converting {split_name} annotations...")
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    labels_dir = Path(output_dir) / "labels" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image ID to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Convert each image's annotations
    converted_count = 0
    for image_id, annotations in tqdm(annotations_by_image.items(), desc=f"Converting {split_name}"):
        filename = id_to_filename[image_id]
        img_width, img_height = id_to_size[image_id]
        
        # Create label file path
        label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = labels_dir / label_filename
        
        # Convert annotations for this image
        yolo_lines = []
        for ann in annotations:
            if 'rotated_bbox' in ann and len(ann['rotated_bbox']) == 8:
                yolo_line = convert_rotated_bbox_to_yolo_obb(
                    ann['rotated_bbox'], img_width, img_height
                )
                yolo_lines.append(yolo_line)
        
        # Write label file
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            converted_count += 1
    
    print(f"Converted {converted_count} images for {split_name} split")


def main():
    """Main conversion function."""
    # Paths
    base_dir = Path("/home/alec/git/pokemon")
    data_dir = base_dir / "data" / "training_100k"
    output_dir = base_dir / "data" / "yolo_obb"
    
    # Create output directory structure
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    
    # Convert each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Source paths
        coco_json = data_dir / "annotations" / f"{split}_annotations.json"
        images_src = data_dir / "images"
        
        # Check if annotation file exists
        if not coco_json.exists():
            print(f"Warning: {coco_json} not found, skipping {split} split")
            continue
        
        # Convert annotations
        convert_coco_to_yolo_obb(
            str(coco_json),
            str(images_src),
            str(output_dir),
            split
        )
        
        # Create symlink to images (to avoid copying large files)
        images_dst = output_dir / "images" / split
        if not images_dst.exists():
            try:
                images_dst.symlink_to(images_src.resolve())
                print(f"Created symlink: {images_dst} -> {images_src}")
            except OSError:
                print(f"Could not create symlink, you may need to copy images manually")
    
    print(f"\nConversion complete! YOLO OBB dataset saved to: {output_dir}")
    print("\nDataset structure:")
    print("├── images/")
    print("│   ├── train/ -> ../../training_100k/images")
    print("│   ├── val/ -> ../../training_100k/images") 
    print("│   └── test/ -> ../../training_100k/images")
    print("└── labels/")
    print("    ├── train/")
    print("    ├── val/")
    print("    └── test/")


if __name__ == "__main__":
    main()
