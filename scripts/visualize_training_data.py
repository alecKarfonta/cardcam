#!/usr/bin/env python3
"""
Visualize generated training data for segmentation.
Shows images with overlaid segmentation masks and bounding boxes.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from typing import List, Dict, Any


def load_coco_annotations(annotation_file: str) -> Dict[str, Any]:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def visualize_image_with_annotations(image_path: str, annotations: List[Dict], 
                                   image_info: Dict, save_path: str = None) -> None:
    """Visualize an image with its annotations."""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Show original image
    ax1.imshow(image_rgb)
    ax1.set_title(f"Original Image\n{Path(image_path).name}")
    ax1.axis('off')
    
    # Show image with annotations
    ax2.imshow(image_rgb)
    ax2.set_title(f"With Annotations ({len(annotations)} cards)")
    ax2.axis('off')
    
    # Colors for different cards
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(annotations), 1)))
    
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        
        # Draw bounding box
        bbox = ann['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Draw segmentation mask
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation'][0]  # First segmentation
            if len(segmentation) >= 6:  # At least 3 points (x,y pairs)
                # Convert to polygon points
                points = np.array(segmentation).reshape(-1, 2)
                polygon = patches.Polygon(
                    points, closed=True, alpha=0.3, 
                    facecolor=color, edgecolor=color, linewidth=1
                )
                ax2.add_patch(polygon)
        
        # Add annotation ID
        ax2.text(bbox[0], bbox[1] - 5, f'Card {i+1}', 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_mask_overlay(image_path: str, annotations: List[Dict]) -> np.ndarray:
    """Create an overlay showing all segmentation masks."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image_rgb)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]
    
    for i, ann in enumerate(annotations):
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation'][0]
            if len(segmentation) >= 6:
                # Convert to mask
                points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                color = colors[i % len(colors)]
                cv2.fillPoly(mask_overlay, [points], color)
    
    # Blend with original image
    alpha = 0.4
    blended = cv2.addWeighted(image_rgb, 1-alpha, mask_overlay, alpha, 0)
    
    return blended


def analyze_dataset_statistics(coco_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze dataset statistics."""
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Basic statistics
    num_images = len(images)
    num_annotations = len(annotations)
    
    # Cards per image
    cards_per_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        cards_per_image[image_id] = cards_per_image.get(image_id, 0) + 1
    
    cards_per_image_list = list(cards_per_image.values())
    
    # Annotation areas
    areas = [ann['area'] for ann in annotations]
    
    # Bounding box sizes
    bbox_widths = [ann['bbox'][2] for ann in annotations]
    bbox_heights = [ann['bbox'][3] for ann in annotations]
    
    stats = {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'avg_cards_per_image': np.mean(cards_per_image_list) if cards_per_image_list else 0,
        'min_cards_per_image': min(cards_per_image_list) if cards_per_image_list else 0,
        'max_cards_per_image': max(cards_per_image_list) if cards_per_image_list else 0,
        'avg_area': np.mean(areas) if areas else 0,
        'avg_bbox_width': np.mean(bbox_widths) if bbox_widths else 0,
        'avg_bbox_height': np.mean(bbox_heights) if bbox_heights else 0,
        'cards_per_image_dist': cards_per_image_list,
        'areas': areas,
        'bbox_widths': bbox_widths,
        'bbox_heights': bbox_heights
    }
    
    return stats


def plot_dataset_statistics(stats: Dict[str, Any], save_path: str = None):
    """Plot dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cards per image distribution
    axes[0, 0].hist(stats['cards_per_image_dist'], bins=range(1, max(stats['cards_per_image_dist'])+2), 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Cards per Image Distribution')
    axes[0, 0].set_xlabel('Number of Cards')
    axes[0, 0].set_ylabel('Frequency')
    
    # Area distribution
    axes[0, 1].hist(stats['areas'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Card Area Distribution')
    axes[0, 1].set_xlabel('Area (pixels²)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Bounding box width distribution
    axes[1, 0].hist(stats['bbox_widths'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Bounding Box Width Distribution')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Bounding box height distribution
    axes[1, 1].hist(stats['bbox_heights'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Bounding Box Height Distribution')
    axes[1, 1].set_xlabel('Height (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize training data")
    parser.add_argument("--data_dir", type=str, default="data/sample_training",
                       help="Directory containing training data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                       help="Data split to visualize")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--show_stats", action="store_true",
                       help="Show dataset statistics")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    annotations_file = data_dir / "annotations" / f"{args.split}_annotations.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from: {annotations_file}")
    coco_data = load_coco_annotations(str(annotations_file))
    
    # Create image ID to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create image ID to image info mapping
    image_info_map = {img['id']: img for img in coco_data['images']}
    
    print(f"Dataset contains {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Visualize samples
    print(f"Visualizing {args.num_samples} samples...")
    
    sample_images = coco_data['images'][:args.num_samples]
    
    for i, image_info in enumerate(sample_images):
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = images_dir / image_filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        annotations = image_annotations.get(image_id, [])
        
        # Create visualization
        output_path = output_dir / f"{args.split}_sample_{i:03d}_visualization.png"
        visualize_image_with_annotations(
            str(image_path), annotations, image_info, str(output_path)
        )
        
        # Create mask overlay
        mask_overlay = create_mask_overlay(str(image_path), annotations)
        mask_output_path = output_dir / f"{args.split}_sample_{i:03d}_masks.png"
        
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_overlay)
        plt.title(f"Segmentation Masks - {image_filename}")
        plt.axis('off')
        plt.savefig(mask_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Processed sample {i+1}/{len(sample_images)}: {image_filename} ({len(annotations)} cards)")
    
    # Show statistics if requested
    if args.show_stats:
        print("\nDataset Statistics:")
        stats = analyze_dataset_statistics(coco_data)
        
        print(f"Number of images: {stats['num_images']}")
        print(f"Number of annotations: {stats['num_annotations']}")
        print(f"Average cards per image: {stats['avg_cards_per_image']:.2f}")
        print(f"Cards per image range: {stats['min_cards_per_image']}-{stats['max_cards_per_image']}")
        print(f"Average card area: {stats['avg_area']:.0f} pixels²")
        print(f"Average bounding box: {stats['avg_bbox_width']:.0f}x{stats['avg_bbox_height']:.0f} pixels")
        
        # Plot statistics
        stats_output_path = output_dir / f"{args.split}_statistics.png"
        plot_dataset_statistics(stats, str(stats_output_path))
    
    print(f"\nVisualization complete! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
