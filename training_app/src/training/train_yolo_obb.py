#!/usr/bin/env python3
"""
YOLO OBB Training Script for Trading Card Detection

This script trains a YOLOv11 model with Oriented Bounding Boxes (OBB) for detecting
trading cards with arbitrary orientations.

Features:
- Single or multi-dataset training
- Combine synthetic and hand-crafted datasets
- Configurable validation ratios (favor hand-crafted data in evaluation)
- Hand-crafted augmentation multipliers (stretch small datasets)
- Enhanced augmentations with albumentations
- MLflow experiment tracking

For multi-dataset usage guide, see docs/multi_dataset_training.md or run:
  python train_yolo_obb.py --multi-dataset-help
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from ultralytics import YOLO
from ultralytics.utils import callbacks
import mlflow
import mlflow.pytorch
from datetime import datetime
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# Try to import albumentations functionality
try:
    from .albumentations_augmentations import (
        OrientationAugmentations, 
        YOLOOBBAugmentations,
        create_card_augmentation_config
    )
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("Albumentations functionality not available. Install albumentations for enhanced augmentations.")

# Force logging to stdout with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_source_datasets(dataset_configs: List[str], output_dir: str, num_samples_per_dataset: int = 8):
    """
    Visualize samples from each source dataset BEFORE merging.
    
    Args:
        dataset_configs: List of paths to dataset YAML configs
        output_dir: Directory to save visualization images
        num_samples_per_dataset: Number of samples to show from each dataset
    """
    try:
        logger.info(f"\nVisualizing {len(dataset_configs)} source datasets...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, config_path in enumerate(dataset_configs):
            logger.info(f"Visualizing dataset {idx}: {config_path}")
            
            # Load config
            config = load_config(config_path)
            dataset_path = Path(config['path'])
            if not dataset_path.is_absolute():
                dataset_path = Path(config_path).parent / dataset_path
            
            train_images_dir = dataset_path / config['train']
            train_labels_dir = dataset_path / "labels" / "train"
            
            # Get images
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(train_images_dir.glob(f"*{ext}")))
            
            if not image_files:
                logger.warning(f"No images found in dataset {idx}")
                continue
            
            # Sample random images
            sampled = random.sample(image_files, min(num_samples_per_dataset, len(image_files)))
            
            # Create visualization
            rows = 2
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
            fig.suptitle(f'Source Dataset {idx}: {Path(config_path).name}\n({len(image_files)} total images)', 
                        fontsize=14, fontweight='bold')
            
            for i, img_path in enumerate(sampled):
                if i >= rows * cols:
                    break
                    
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display image
                ax.imshow(image)
                ax.set_title(f'{img_path.name}\n{image.shape[1]}x{image.shape[0]}', fontsize=8)
                ax.axis('off')
                
                # Load and draw labels
                label_path = train_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        num_boxes = 0
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                num_boxes += 1
                                coords = list(map(float, parts[1:]))
                                img_h, img_w = image.shape[:2]
                                
                                if len(coords) >= 8:
                                    # OBB format
                                    points = []
                                    for j in range(0, 8, 2):
                                        x = coords[j] * img_w
                                        y = coords[j+1] * img_h
                                        points.append([x, y])
                                    polygon = Polygon(points, linewidth=2, edgecolor='cyan', facecolor='none')
                                    ax.add_patch(polygon)
                                else:
                                    # Regular bbox
                                    x_center, y_center, width, height = coords[:4]
                                    x_center *= img_w
                                    y_center *= img_h
                                    width *= img_w
                                    height *= img_h
                                    x1 = x_center - width / 2
                                    y1 = y_center - height / 2
                                    rect = patches.Rectangle(
                                        (x1, y1), width, height,
                                        linewidth=2, edgecolor='cyan', facecolor='none'
                                    )
                                    ax.add_patch(rect)
                        
                        # Update title with box count
                        title = ax.get_title()
                        ax.set_title(f'{title}\n{num_boxes} boxes', fontsize=8)
            
            # Save visualization for this dataset
            viz_path = output_path / f"source_dataset_{idx}_{Path(config_path).stem}.png"
            plt.tight_layout()
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved visualization: {viz_path}")
            
    except Exception as e:
        logger.error(f"Error visualizing source datasets: {e}", exc_info=True)


def process_single_image_with_augmentations(args):
    """
    Worker function for parallel processing of image augmentations.
    
    Args:
        args: Tuple of (img_file, label_file, multiplier, dataset_name, train_images_dir, train_labels_dir, apply_pre_augmentation)
    
    Returns:
        Tuple of (success_count, error_message)
    """
    img_file, label_file, multiplier, dataset_name, train_images_dir, train_labels_dir, apply_pre_augmentation = args
    
    success_count = 0
    errors = []
    
    try:
        # Load image once if pre-augmentation is enabled
        source_image = None
        if apply_pre_augmentation and multiplier > 1:
            source_image = cv2.imread(str(img_file))
            if source_image is None:
                return (0, f"Failed to load {img_file}")
        
        # Create all augmented versions
        for copy_idx in range(multiplier):
            if multiplier > 1:
                new_stem = f"{img_file.stem}_aug{copy_idx}_{dataset_name}"
            else:
                new_stem = f"{img_file.stem}_{dataset_name}"
            
            # Save image (with pre-augmentation if enabled)
            dst_img = Path(train_images_dir) / f"{new_stem}{img_file.suffix}"
            try:
                if apply_pre_augmentation and multiplier > 1 and source_image is not None:
                    # Apply deterministic augmentation
                    augmented_image = apply_deterministic_augmentation(source_image, copy_idx, multiplier)
                    cv2.imwrite(str(dst_img), augmented_image)
                else:
                    # Just copy the original
                    shutil.copy2(img_file, dst_img)
                success_count += 1
            except Exception as e:
                errors.append(f"Failed to save image {dst_img}: {e}")
                continue
            
            # Copy label (same for all duplicates)
            dst_label = Path(train_labels_dir) / f"{new_stem}.txt"
            try:
                shutil.copy2(label_file, dst_label)
            except Exception as e:
                errors.append(f"Failed to copy label {dst_label}: {e}")
                continue
        
        return (success_count, None if not errors else "; ".join(errors))
    
    except Exception as e:
        return (0, f"Error processing {img_file}: {e}")


def apply_deterministic_augmentation(image: np.ndarray, aug_index: int, total_augs: int) -> np.ndarray:
    """
    Apply deterministic PHOTOMETRIC augmentation to an image based on aug_index.
    This ensures each duplicate gets a different, reproducible augmentation.
    
    IMPORTANT: Only applies color/lighting augmentations that don't affect bounding boxes.
    Geometric augmentations (rotation, scaling, etc.) are handled by YOLO during training.
    
    Args:
        image: Input image (numpy array)
        aug_index: Index of the augmentation (0 to total_augs-1)
        total_augs: Total number of augmentation variants
    
    Returns:
        Augmented image
    """
    if aug_index == 0:
        # First copy is always original
        return image.copy()
    
    # Set seed for reproducibility of this augmentation variant
    np.random.seed(aug_index * 12345)
    
    # Apply different augmentations based on aug_index modulo patterns
    augmented = image.copy()
    
    # ========== PHOTOMETRIC AUGMENTATIONS ONLY ==========
    # These do NOT change bounding box coordinates
    
    # Brightness adjustment (cycle through different levels)
    brightness_factor = 0.7 + (aug_index % 7) * 0.1  # 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
    augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
    
    # Contrast adjustment (alternate every few)
    if aug_index % 3 == 1:
        contrast_factor = 1.2
        augmented = np.clip((augmented - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
    elif aug_index % 3 == 2:
        contrast_factor = 0.8
        augmented = np.clip((augmented - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
    
    # Gaussian noise (every 4th image)
    if aug_index % 4 == 0 and aug_index > 0:
        noise = np.random.normal(0, 10, augmented.shape).astype(np.int16)
        augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Gaussian blur (every 5th image)
    if aug_index % 5 == 0 and aug_index > 0:
        augmented = cv2.GaussianBlur(augmented, (5, 5), 0)
    
    # HSV color shifts (more variations)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Hue shift (cycle through color temperatures)
    if aug_index % 8 == 1:
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180  # Warm shift
    elif aug_index % 8 == 2:
        hsv[:, :, 0] = (hsv[:, :, 0] - 10) % 180  # Cool shift
    
    # Saturation adjustment
    if aug_index % 7 == 1:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # More saturated
    elif aug_index % 7 == 2:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)  # Less saturated
    
    augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Gamma correction (different lighting conditions)
    if aug_index % 6 == 4:
        gamma = 1.2
        augmented = np.clip(255 * (augmented / 255) ** (1/gamma), 0, 255).astype(np.uint8)
    elif aug_index % 6 == 5:
        gamma = 0.8
        augmented = np.clip(255 * (augmented / 255) ** (1/gamma), 0, 255).astype(np.uint8)
    
    # Reset random seed
    np.random.seed(None)
    
    return augmented


def merge_datasets(
    dataset_configs: List[str],
    output_dir: str,
    eval_handcrafted_ratio: float = 0.8,
    handcrafted_aug_multiplier: int = 1,
    handcrafted_dataset_index: int = -1,
    apply_pre_augmentation: bool = False,
) -> str:
    """
    Merge multiple datasets into a single combined dataset.
    
    Args:
        dataset_configs: List of paths to dataset YAML configs
        output_dir: Directory to save merged dataset
        eval_handcrafted_ratio: Ratio of hand-crafted data in validation set (0.0-1.0)
        handcrafted_aug_multiplier: How many times to duplicate hand-crafted training data
        handcrafted_dataset_index: Index of hand-crafted dataset in the list (default: -1 = last)
        apply_pre_augmentation: Apply deterministic augmentations to each duplicate
    
    Returns:
        Path to merged dataset config file
    """
    print(f"\n{'='*60}", flush=True)
    print(f"MERGING {len(dataset_configs)} DATASETS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Hand-crafted augmentation multiplier: {handcrafted_aug_multiplier}x", flush=True)
    if apply_pre_augmentation:
        print("Pre-augmentation: ENABLED (each duplicate gets deterministic augmentations)", flush=True)
    else:
        print("Pre-augmentation: DISABLED (duplicates are identical, will get random augmentations during training)", flush=True)
    
    logger.info(f"Merging {len(dataset_configs)} datasets...")
    logger.info(f"Hand-crafted augmentation multiplier: {handcrafted_aug_multiplier}x")
    if apply_pre_augmentation:
        logger.info("Pre-augmentation: ENABLED (each duplicate gets deterministic augmentations)")
    else:
        logger.info("Pre-augmentation: DISABLED (duplicates are identical, will get random augmentations during training)")
    
    if len(dataset_configs) < 2:
        logger.warning("Only one dataset provided, no merging needed")
        return dataset_configs[0]
    
    # OVERSAMPLING STRATEGY EXPLANATION:
    # By creating N duplicates of each hand-crafted image, we effectively "oversample" the
    # hand-crafted dataset. During training, YOLO will see hand-crafted images N times more
    # often than synthetic images (per epoch). Combined with YOLO's random augmentation
    # pipeline, each duplicate will get different augmentations every epoch, creating
    # massive diversity even from a small hand-crafted dataset.
    #
    # For example, with 300 hand-crafted images and 50x multiplier:
    # - 15,000 hand-crafted training samples (300 x 50)
    # - Each epoch, model sees each hand-crafted image 50x with different augmentations
    # - Over 50 epochs, that's 2,500 unique augmented views per original image!
    # - This is equivalent to having 750,000 synthetic variations from 300 originals
    
    # Visualize source datasets before merging
    logger.info("\n" + "="*60)
    logger.info("VISUALIZING SOURCE DATASETS (before merging)")
    logger.info("="*60)
    source_viz_dir = Path(output_dir) / "source_datasets_visualization"
    try:
        visualize_source_datasets(dataset_configs, str(source_viz_dir), num_samples_per_dataset=8)
    except Exception as e:
        logger.warning(f"Failed to visualize source datasets: {e}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_images_dir = output_path / "images" / "train"
    train_labels_dir = output_path / "labels" / "train"
    val_images_dir = output_path / "images" / "val"
    val_labels_dir = output_path / "labels" / "val"
    
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load all dataset configs
    configs = []
    for config_path in dataset_configs:
        with open(config_path, 'r') as f:
            configs.append(yaml.safe_load(f))
    
    # Identify hand-crafted dataset
    if handcrafted_dataset_index < 0:
        handcrafted_dataset_index = len(configs) + handcrafted_dataset_index
    
    # Track statistics
    stats = defaultdict(lambda: {'train_images': 0, 'train_labels': 0, 'val_images': 0, 'val_labels': 0})
    
    # First pass: collect all validation image lists and counts
    validation_files_by_dataset = {}
    for idx, (config_path, config) in enumerate(zip(dataset_configs, configs)):
        is_handcrafted = (idx == handcrafted_dataset_index)
        dataset_name = f"dataset_{idx}" + ("_handcrafted" if is_handcrafted else "_synthetic")
        
        source_base = Path(config['path'])
        if not source_base.is_absolute():
            source_base = Path(config_path).parent / source_base
        
        source_val_images = source_base / config['val']
        if source_val_images.exists():
            val_image_files = list(source_val_images.glob("*.jpg")) + \
                             list(source_val_images.glob("*.png")) + \
                             list(source_val_images.glob("*.jpeg"))
            validation_files_by_dataset[idx] = {
                'files': val_image_files,
                'is_handcrafted': is_handcrafted,
                'dataset_name': dataset_name,
                'source_base': source_base,
                'config': config
            }
    
    # Calculate target counts for validation set to achieve desired ratio
    total_handcrafted_val = sum(len(d['files']) for d in validation_files_by_dataset.values() if d['is_handcrafted'])
    total_synthetic_val = sum(len(d['files']) for d in validation_files_by_dataset.values() if not d['is_handcrafted'])
    
    print(f"\n{'='*60}", flush=True)
    print(f"VALIDATION SET COMPOSITION", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Available hand-crafted validation images: {total_handcrafted_val}", flush=True)
    print(f"Available synthetic validation images: {total_synthetic_val}", flush=True)
    print(f"Target hand-crafted ratio: {eval_handcrafted_ratio:.1%}", flush=True)
    
    logger.info(f"\n" + "="*60)
    logger.info(f"VALIDATION SET COMPOSITION")
    logger.info(f"="*60)
    logger.info(f"Available hand-crafted validation images: {total_handcrafted_val}")
    logger.info(f"Available synthetic validation images: {total_synthetic_val}")
    logger.info(f"Target hand-crafted ratio: {eval_handcrafted_ratio:.1%}")
    
    # Calculate how many samples we need from each category
    # Target: handcrafted_samples / (handcrafted_samples + synthetic_samples) = eval_handcrafted_ratio
    # If we take all available, calculate what ratio we'd get
    if total_handcrafted_val + total_synthetic_val > 0:
        # Use a reasonable total size (take all from smaller set, sample from larger)
        if eval_handcrafted_ratio >= 1.0:
            target_handcrafted_samples = total_handcrafted_val
            target_synthetic_samples = 0
        elif eval_handcrafted_ratio <= 0.0:
            target_handcrafted_samples = 0
            target_synthetic_samples = total_synthetic_val
        else:
            # Calculate samples to achieve ratio
            # If handcrafted is smaller, take all handcrafted and calculate synthetic
            if total_handcrafted_val * (1 - eval_handcrafted_ratio) <= total_synthetic_val * eval_handcrafted_ratio:
                target_handcrafted_samples = total_handcrafted_val
                target_synthetic_samples = int(target_handcrafted_samples * (1 - eval_handcrafted_ratio) / eval_handcrafted_ratio)
            else:
                # Synthetic is smaller, take all synthetic and calculate handcrafted
                target_synthetic_samples = total_synthetic_val
                target_handcrafted_samples = int(target_synthetic_samples * eval_handcrafted_ratio / (1 - eval_handcrafted_ratio))
            
            # Ensure we don't exceed available samples
            target_handcrafted_samples = min(target_handcrafted_samples, total_handcrafted_val)
            target_synthetic_samples = min(target_synthetic_samples, total_synthetic_val)
    else:
        target_handcrafted_samples = 0
        target_synthetic_samples = 0
    
    print(f"\nValidation sampling targets:", flush=True)
    print(f"  Hand-crafted: {target_handcrafted_samples} images", flush=True)
    print(f"  Synthetic: {target_synthetic_samples} images", flush=True)
    print(f"  Total: {target_handcrafted_samples + target_synthetic_samples} images", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    logger.info(f"Validation sampling targets:")
    logger.info(f"  Hand-crafted: {target_handcrafted_samples} images")
    logger.info(f"  Synthetic: {target_synthetic_samples} images")
    logger.info(f"  Total: {target_handcrafted_samples + target_synthetic_samples} images")
    logger.info(f"="*60 + "\n")
    
    # Process each dataset
    for idx, (config_path, config) in enumerate(zip(dataset_configs, configs)):
        is_handcrafted = (idx == handcrafted_dataset_index)
        dataset_name = f"dataset_{idx}" + ("_handcrafted" if is_handcrafted else "_synthetic")
        
        logger.info(f"\n{'-'*60}")
        logger.info(f"Processing {dataset_name} from {config_path}")
        logger.info(f"Type: {'HAND-CRAFTED' if is_handcrafted else 'SYNTHETIC'}")
        if is_handcrafted:
            logger.info(f"Augmentation multiplier: {handcrafted_aug_multiplier}x")
        
        # Get source paths
        source_base = Path(config['path'])
        if not source_base.is_absolute():
            source_base = Path(config_path).parent / source_base
        
        # Process training data
        source_train_images = source_base / config['train']
        source_train_labels = source_base / "labels" / "train"
        
        if source_train_images.exists():
            image_files = list(source_train_images.glob("*.jpg")) + \
                         list(source_train_images.glob("*.png")) + \
                         list(source_train_images.glob("*.jpeg"))
            
            logger.info(f"Found {len(image_files)} training images")
            
            # For hand-crafted data, duplicate images to stretch the dataset
            multiplier = handcrafted_aug_multiplier if is_handcrafted else 1
            if multiplier > 1:
                logger.info(f"Will create {multiplier} copies of each image (total: {len(image_files) * multiplier} images)")
                if apply_pre_augmentation and is_handcrafted:
                    logger.info(f"Pre-augmentation ENABLED - each duplicate will have deterministic augmentations applied")
                else:
                    logger.info(f"Pre-augmentation DISABLED - duplicates are exact copies (will get different augmentations during training)")
            
            # Prepare list of images with labels for processing
            image_label_pairs = []
            for img_file in image_files:
                if not img_file.exists():
                    continue
                label_file = source_train_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    image_label_pairs.append((img_file, label_file))
            
            total_to_process = len(image_label_pairs)
            print(f"\nProcessing {total_to_process} training images from {dataset_name}...", flush=True)
            
            # Use parallel processing for hand-crafted data with multiplier > 1
            if is_handcrafted and multiplier > 1 and apply_pre_augmentation:
                # Parallel processing for augmentation
                num_workers = max(1, cpu_count() - 1)  # Leave one core free
                print(f"Using {num_workers} parallel workers for pre-augmentation...", flush=True)
                
                # Prepare arguments for parallel processing
                worker_args = [
                    (img_file, label_file, multiplier, dataset_name, train_images_dir, train_labels_dir, apply_pre_augmentation)
                    for img_file, label_file in image_label_pairs
                ]
                
                # Process in parallel with progress tracking
                with Pool(processes=num_workers) as pool:
                    processed_count = 0
                    for result in pool.imap_unordered(process_single_image_with_augmentations, worker_args):
                        processed_count += 1
                        success_count, error_msg = result
                        
                        stats[dataset_name]['train_images'] += success_count
                        stats[dataset_name]['train_labels'] += success_count
                        
                        if error_msg:
                            logger.warning(error_msg)
                        
                        # Progress update
                        if processed_count % 5 == 0 or processed_count == total_to_process:
                            percent = processed_count * 100 // total_to_process
                            print(f"  [{processed_count}/{total_to_process}] {percent}% complete", flush=True)
                
                print(f"Completed processing {total_to_process} images with {multiplier}x augmentation", flush=True)
            
            else:
                # Sequential processing for synthetic data or simple copies
                for idx, (img_file, label_file) in enumerate(image_label_pairs, 1):
                    if idx % 1000 == 0:
                        print(f"  Progress: {idx}/{total_to_process} ({idx*100//total_to_process}%)", flush=True)
                    
                    for copy_idx in range(multiplier):
                        if multiplier > 1:
                            new_stem = f"{img_file.stem}_aug{copy_idx}_{dataset_name}"
                        else:
                            new_stem = f"{img_file.stem}_{dataset_name}"
                        
                        # Copy image
                        dst_img = train_images_dir / f"{new_stem}{img_file.suffix}"
                        try:
                            shutil.copy2(img_file, dst_img)
                            stats[dataset_name]['train_images'] += 1
                        except Exception as e:
                            logger.error(f"Failed to copy image: {e}")
                            continue
                        
                        # Copy label
                        dst_label = train_labels_dir / f"{new_stem}.txt"
                        try:
                            shutil.copy2(label_file, dst_label)
                            stats[dataset_name]['train_labels'] += 1
                        except Exception as e:
                            logger.error(f"Failed to copy label: {e}")
                            continue
        
        # Process validation data with ratio-based sampling
        print(f"\n{'='*60}", flush=True)
        print(f"Processing validation data for {dataset_name}...", flush=True)
        print(f"Dataset index {idx} in validation_files_by_dataset: {idx in validation_files_by_dataset}", flush=True)
        
        logger.info(f"\nProcessing validation data for {dataset_name}...")
        logger.info(f"Available validation files for dataset {idx}: {len(validation_files_by_dataset.get(idx, {}).get('files', []))}")
        
        if idx in validation_files_by_dataset:
            val_data = validation_files_by_dataset[idx]
            val_image_files = val_data['files']
            source_val_labels = source_base / "labels" / "val"
            
            logger.info(f"Found {len(val_image_files)} validation images in {source_base / config['val']}")
            
            # Calculate how many samples to take from this dataset
            if is_handcrafted:
                # Distribute handcrafted target among handcrafted datasets
                num_handcrafted_datasets = sum(1 for d in validation_files_by_dataset.values() if d['is_handcrafted'])
                my_target = target_handcrafted_samples // max(num_handcrafted_datasets, 1)
                logger.info(f"Hand-crafted validation target: {my_target} (from total target: {target_handcrafted_samples})")
            else:
                # Distribute synthetic target among synthetic datasets  
                num_synthetic_datasets = sum(1 for d in validation_files_by_dataset.values() if not d['is_handcrafted'])
                my_target = target_synthetic_samples // max(num_synthetic_datasets, 1)
                logger.info(f"Synthetic validation target: {my_target} (from total target: {target_synthetic_samples})")
            
            # Don't exceed available images
            num_samples = min(my_target, len(val_image_files))
            
            # Take at least 1 if available
            if len(val_image_files) > 0:
                num_samples = max(1, num_samples)
            
            print(f"Will sample {num_samples} validation images from {dataset_name}", flush=True)
            logger.info(f"Will sample {num_samples} validation images from {dataset_name}")
            
            # Sample validation images randomly
            sampled_files = random.sample(val_image_files, num_samples) if num_samples > 0 else []
            
            print(f"Sampled {len(sampled_files)} files, now copying...", flush=True)
            
            for img_file in sampled_files:
                label_file = source_val_labels / f"{img_file.stem}.txt"
                new_stem = f"{img_file.stem}_{dataset_name}"
                
                # Copy image
                dst_img = val_images_dir / f"{new_stem}{img_file.suffix}"
                shutil.copy2(img_file, dst_img)
                stats[dataset_name]['val_images'] += 1
                
                # Copy label if exists
                if label_file.exists():
                    dst_label = val_labels_dir / f"{new_stem}.txt"
                    shutil.copy2(label_file, dst_label)
                    stats[dataset_name]['val_labels'] += 1
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("MERGED DATASET STATISTICS")
    logger.info("="*60)
    for dataset_name, counts in stats.items():
        logger.info(f"\n{dataset_name}:")
        logger.info(f"  Training:   {counts['train_images']} images, {counts['train_labels']} labels")
        logger.info(f"  Validation: {counts['val_images']} images, {counts['val_labels']} labels")
    
    total_train = sum(s['train_images'] for s in stats.values())
    total_val = sum(s['val_images'] for s in stats.values())
    logger.info(f"\nTOTAL:")
    logger.info(f"  Training:   {total_train} images")
    logger.info(f"  Validation: {total_val} images")
    
    # Calculate actual validation ratio
    handcrafted_name = f"dataset_{handcrafted_dataset_index}_handcrafted"
    if handcrafted_name in stats:
        actual_ratio = stats[handcrafted_name]['val_images'] / max(total_val, 1)
        logger.info(f"\nActual hand-crafted validation ratio: {actual_ratio:.2%}")
    logger.info("="*60 + "\n")
    
    # Create merged config file based on first dataset's config
    merged_config = configs[0].copy()
    merged_config['path'] = str(output_path.absolute())
    merged_config['train'] = "images/train"
    merged_config['val'] = "images/val"
    
    # Add metadata about merging
    merged_config['merged_dataset'] = {
        'source_datasets': dataset_configs,
        'handcrafted_dataset_index': handcrafted_dataset_index,
        'eval_handcrafted_ratio': eval_handcrafted_ratio,
        'handcrafted_aug_multiplier': handcrafted_aug_multiplier,
        'apply_pre_augmentation': apply_pre_augmentation,
        'merge_date': datetime.now().isoformat(),
        'statistics': dict(stats)
    }
    
    # Save merged config
    merged_config_path = output_path / "merged_dataset.yaml"
    with open(merged_config_path, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Merged dataset config saved to: {merged_config_path}")
    
    # Visualize merged dataset samples
    logger.info("\n" + "="*60)
    logger.info("Visualizing merged dataset samples...")
    logger.info("="*60)
    merged_viz_dir = output_path / "merged_dataset_visualization"
    try:
        visualize_raw_dataset_samples(
            str(merged_config_path),
            str(merged_viz_dir),
            num_examples=16
        )
    except Exception as e:
        logger.warning(f"Failed to visualize merged dataset: {e}")
    
    return str(merged_config_path)


def visualize_raw_dataset_samples(dataset_config: str, output_dir: str, num_examples: int = 16):
    """
    Visualize RAW dataset samples WITHOUT any augmentation to verify data quality.
    
    Args:
        dataset_config: Path to dataset YAML configuration
        output_dir: Directory to save visualization images
        num_examples: Number of examples to visualize
    """
    try:
        logger.info(f"Visualizing {num_examples} RAW dataset samples (no augmentation)...")
        
        # Load dataset configuration
        config = load_config(dataset_config)
        dataset_path = Path(config['path'])
        train_images_dir = dataset_path / config['train']
        train_labels_dir = dataset_path / "labels" / "train"
        
        # Get list of training images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(train_images_dir.glob(f"*{ext}")))
        
        if not image_files:
            logger.error("No training images found!")
            return
        
        logger.info(f"Found {len(image_files)} training images")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample random images
        sampled_images = random.sample(image_files, min(num_examples, len(image_files)))
        
        # Create grid visualization
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(24, 24))
        fig.suptitle('RAW Training Data Samples (NO Augmentation)', fontsize=16, fontweight='bold')
        
        for i, img_path in enumerate(sampled_images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load labels
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            # Display image
            ax.imshow(image)
            ax.set_title(f'{img_path.name}\n{image.shape[1]}x{image.shape[0]}', fontsize=8)
            ax.axis('off')
            
            # Draw bounding boxes
            if label_path.exists():
                with open(label_path, 'r') as f:
                    num_boxes = 0
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            num_boxes += 1
                            class_id = int(parts[0])
                            # YOLO OBB format: class x_center y_center width height [x1 y1 x2 y2 x3 y3 x4 y4]
                            coords = list(map(float, parts[1:]))
                            
                            img_h, img_w = image.shape[:2]
                            
                            # Check if OBB format (8 coordinates) or regular bbox (4 coordinates)
                            if len(coords) >= 8:
                                # OBB format - draw polygon
                                points = []
                                for j in range(0, 8, 2):
                                    x = coords[j] * img_w
                                    y = coords[j+1] * img_h
                                    points.append([x, y])
                                polygon = Polygon(points, linewidth=2, edgecolor='red', facecolor='none')
                                ax.add_patch(polygon)
                            else:
                                # Regular bbox format
                                x_center, y_center, width, height = coords[:4]
                                x_center *= img_w
                                y_center *= img_h
                                width *= img_w
                                height *= img_h
                                
                                x1 = x_center - width / 2
                                y1 = y_center - height / 2
                                
                                rect = patches.Rectangle(
                                    (x1, y1), width, height,
                                    linewidth=2, edgecolor='red', facecolor='none'
                                )
                                ax.add_patch(rect)
                    
                    # Add box count to title
                    title = ax.get_title()
                    ax.set_title(f'{title}\n{num_boxes} boxes', fontsize=8)
            else:
                logger.warning(f"No label file for: {img_path.name}")
        
        # Save visualization
        viz_path = output_path / "raw_dataset_samples.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RAW dataset samples saved to: {viz_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing raw dataset samples: {e}", exc_info=True)


def visualize_augmented_examples(dataset_config: str, output_dir: str, num_examples: int = 8, use_albumentations: bool = True, debug: bool = False):
    """
    Generate and save visualization of augmented training examples.
    
    Args:
        dataset_config: Path to dataset YAML configuration
        output_dir: Directory to save visualization images
        num_examples: Number of examples to generate
        use_albumentations: Whether to use enhanced augmentations
    """
    try:
        # Load dataset configuration
        config = load_config(dataset_config)
        if use_albumentations:
            config = setup_enhanced_augmentations(config, use_albumentations)
        
        # Get dataset paths
        dataset_path = Path(config['path'])
        train_images_dir = dataset_path / config['train']
        train_labels_dir = dataset_path / "labels" / "train"
        
        # Get list of training images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(train_images_dir.glob(f"*{ext}")))
        
        if not image_files:
            print("❌ No training images found!")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create augmentation pipeline if available
        aug_pipeline = None
        if use_albumentations and ALBUMENTATIONS_AVAILABLE:
            from .albumentations_augmentations import OrientationAugmentations
            aug_creator = OrientationAugmentations()
            aug_pipeline = aug_creator.get_pipeline("medium")
        
        print(f"🎨 Generating {num_examples} augmented training examples...")
        
        # Generate examples
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Augmented Training Examples', fontsize=16, fontweight='bold')
        
        for i in range(min(num_examples, 8)):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Select random image
            img_path = random.choice(image_files)
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load labels
            bboxes = []
            class_labels = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # Convert YOLO format to pixel coordinates
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            img_h, img_w = image.shape[:2]
                            x_center *= img_w
                            y_center *= img_h
                            width *= img_w
                            height *= img_h
                            
                            # Convert to corner format for visualization
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            bboxes.append([x1, y1, x2, y2])
                            class_labels.append('card')
            
            # Apply augmentations
            if aug_pipeline and bboxes:
                try:
                    augmented = aug_pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                except Exception as e:
                    print(f"⚠️  Augmentation failed for {img_path.name}: {e}")
                    aug_image = image
                    aug_bboxes = bboxes
                    aug_labels = class_labels
            else:
                aug_image = image
                aug_bboxes = bboxes
                aug_labels = class_labels
            
            # Display image
            ax.imshow(aug_image)
            ax.set_title(f'Example {i+1}: {img_path.name}', fontsize=10)
            ax.axis('off')
            
            # Draw bounding boxes
            for bbox in aug_bboxes:
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
        
        # Save visualization
        viz_path = output_path / "augmented_examples.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Augmented examples saved to: {viz_path}")
        
        # Generate individual detailed examples
        print("🔍 Generating detailed individual examples...")
        
        for i in range(min(4, len(image_files))):
            img_path = random.choice(image_files)
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            # Load and process image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create before/after comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Augmentation Comparison: {img_path.name}', fontsize=14, fontweight='bold')
            
            # Original image
            ax1.imshow(image)
            ax1.set_title('Original', fontsize=12)
            ax1.axis('off')
            
            # Apply augmentation
            if aug_pipeline:
                try:
                    augmented = aug_pipeline(image=image)
                    aug_image = augmented['image']
                except:
                    aug_image = image
            else:
                aug_image = image
            
            # Augmented image
            ax2.imshow(aug_image)
            ax2.set_title('Augmented', fontsize=12)
            ax2.axis('off')
            
            # Save individual comparison
            detail_path = output_path / f"detailed_example_{i+1}.png"
            plt.tight_layout()
            plt.savefig(detail_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Detailed examples saved to: {output_path}")
        
        # Generate augmentation statistics
        _generate_augmentation_stats(config, output_path)
        
    except Exception as e:
        print(f"❌ Error generating augmented examples: {e}")


def _generate_augmentation_stats(config: Dict[str, Any], output_dir: Path):
    """Generate and save augmentation statistics."""
    try:
        aug_config = config.get('augmentation', {})
        
        stats_text = "🎨 AUGMENTATION CONFIGURATION SUMMARY\n"
        stats_text += "=" * 50 + "\n\n"
        
        # Geometric augmentations
        stats_text += "GEOMETRIC AUGMENTATIONS:\n"
        stats_text += f"  Rotation: ±{aug_config.get('degrees', 0)}°\n"
        stats_text += f"  Translation: ±{aug_config.get('translate', 0)*100:.1f}%\n"
        stats_text += f"  Scale: ±{aug_config.get('scale', 0)*100:.1f}%\n"
        stats_text += f"  Shear: ±{aug_config.get('shear', 0)}°\n"
        stats_text += f"  Perspective: {aug_config.get('perspective', 0)}\n\n"
        
        # Flip augmentations
        stats_text += "FLIP AUGMENTATIONS:\n"
        stats_text += f"  Horizontal Flip: {aug_config.get('fliplr', 0)*100:.1f}%\n"
        stats_text += f"  Vertical Flip: {aug_config.get('flipud', 0)*100:.1f}%\n\n"
        
        # Color augmentations
        stats_text += "COLOR AUGMENTATIONS:\n"
        stats_text += f"  Hue Shift: ±{aug_config.get('hsv_h', 0)*100:.1f}%\n"
        stats_text += f"  Saturation: ±{aug_config.get('hsv_s', 0)*100:.1f}%\n"
        stats_text += f"  Brightness: ±{aug_config.get('hsv_v', 0)*100:.1f}%\n\n"
        
        # Advanced augmentations
        stats_text += "ADVANCED AUGMENTATIONS:\n"
        stats_text += f"  Mosaic: {aug_config.get('mosaic', 0)*100:.1f}%\n"
        stats_text += f"  MixUp: {aug_config.get('mixup', 0)*100:.1f}%\n"
        stats_text += f"  Copy-Paste: {aug_config.get('copy_paste', 0)*100:.1f}%\n"
        stats_text += f"  Blur: {aug_config.get('blur', 0)*100:.1f}%\n"
        stats_text += f"  Noise: {aug_config.get('noise', 0)*100:.1f}%\n"
        stats_text += f"  Erasing: {aug_config.get('erasing', 0)*100:.1f}%\n\n"
        
        # Albumentations specific
        alb_config = config.get('albumentations', {})
        if alb_config.get('enabled', False):
            stats_text += "ALBUMENTATIONS SETTINGS:\n"
            stats_text += f"  Enabled: {alb_config.get('enabled', False)}\n"
            stats_text += f"  Intensity: {alb_config.get('intensity', 'medium')}\n"
            stats_text += f"  Orientation Focus: {alb_config.get('orientation_focus', False)}\n"
            stats_text += f"  Max Rotation: {alb_config.get('max_rotation', 60)}°\n"
        
        # Save stats
        stats_path = output_dir / "augmentation_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(stats_text)
        
        print(f"📊 Augmentation stats saved to: {stats_path}")
        
    except Exception as e:
        print(f"⚠️  Could not generate augmentation stats: {e}")


def setup_enhanced_augmentations(config: Dict[str, Any], use_albumentations: bool = True) -> Dict[str, Any]:
    """Setup enhanced augmentation configuration."""
    
    if not use_albumentations or not ALBUMENTATIONS_AVAILABLE:
        logger.info("Using standard YOLO augmentations")
        return config
    
    # Check if albumentations is enabled in config
    albumentations_config = config.get('albumentations', {})
    
    if albumentations_config.get('enabled', False):
        logger.info("Using enhanced albumentations pipeline")
        
        # Get augmentation configuration from our pipeline
        enhanced_config = create_card_augmentation_config()
        
        # Override with albumentations-specific settings
        if albumentations_config.get('orientation_focus', False):
            enhanced_config.update({
                'degrees': albumentations_config.get('max_rotation', 60),
                'shear': albumentations_config.get('shear_limit', 20),
                'perspective': albumentations_config.get('perspective_scale', 0.0005),
            })
        
        # Update the config
        config['augmentation'].update(enhanced_config)
        
        logger.info(f"Enhanced augmentation settings:")
        for key, value in enhanced_config.items():
            logger.info(f"  {key}: {value}")
    
    return config


def setup_mlflow(config: Dict[str, Any]) -> None:
    """Setup MLflow experiment tracking."""
    experiment_name = config['logging']['project']
    run_name = f"{config['logging']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set MLflow tracking URI (local by default)
    mlflow.set_tracking_uri("file:../outputs/mlruns")
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Could not setup MLflow: {e}")
        return
    
    # Start MLflow run
    mlflow.start_run(run_name=run_name)
    
    # Log configuration
    mlflow.log_params({
        "model_type": config['model']['type'],
        "image_size": config['train_params']['image_size'],
        "batch_size": config['train_params']['batch_size'],
        "learning_rate": config['hyperparameters']['lr0'],
        "epochs": config.get('epochs', 100),
    })


def validate_dataset(dataset_config: str) -> bool:
    """Validate that the dataset exists and is properly formatted."""
    try:
        with open(dataset_config, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_path = Path(dataset_config).parent / config['path']
        
        # Check if dataset directories exist
        for split in ['train', 'val']:
            images_path = dataset_path / config[split]
            labels_path = dataset_path / "labels" / split
            
            if not images_path.exists():
                print(f"Error: Images directory not found: {images_path}")
                return False
            
            if not labels_path.exists():
                print(f"Error: Labels directory not found: {labels_path}")
                return False
            
            # Check if there are files in the directories
            image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
            label_files = list(labels_path.glob("*.txt"))
            
            if len(image_files) == 0:
                print(f"Error: No image files found in {images_path}")
                return False
            
            if len(label_files) == 0:
                print(f"Error: No label files found in {labels_path}")
                return False
            
            print(f"✓ {split}: {len(image_files)} images, {len(label_files)} labels")
        
        return True
    
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return False


def create_periodic_validation_callback(dataset_config: str, output_base_dir: str, interval: int = 10, yolo_wrapper=None):
    """
    Create a callback that generates validation predictions periodically during training.
    
    Args:
        dataset_config: Path to dataset YAML configuration
        output_base_dir: Base directory to save visualization images
        interval: Generate visualization every N epochs
        yolo_wrapper: The YOLO wrapper object (not the underlying model)
    
    Returns:
        Callback function
    """
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        epoch = trainer.epoch + 1
        
        # Only generate visualizations at specified intervals
        if epoch % interval != 0:
            return
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}: Generating validation predictions...")
            logger.info(f"{'='*60}")
            
            output_dir = Path(output_base_dir) / f"epoch_{epoch:03d}_predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the YOLO wrapper instead of trainer.model
            visualize_validation_predictions(
                model=yolo_wrapper,
                dataset_config=dataset_config,
                output_dir=str(output_dir),
                num_samples=16,
                conf_threshold=0.25
            )
            
            logger.info(f"Epoch {epoch}: Validation predictions saved to {output_dir}")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error generating periodic validation predictions: {e}", exc_info=True)
    
    return on_train_epoch_end


def visualize_validation_predictions(model, dataset_config: str, output_dir: str, num_samples: int = 16, conf_threshold: float = 0.25):
    """
    Run model predictions on validation samples and visualize results with bounding boxes.
    
    Args:
        model: Trained YOLO model
        dataset_config: Path to dataset YAML configuration
        output_dir: Directory to save visualization images
        num_samples: Number of validation samples to visualize
        conf_threshold: Confidence threshold for predictions
    """
    try:
        logger.info(f"Generating validation predictions visualization...")
        
        # Load dataset configuration
        config = load_config(dataset_config)
        dataset_path = Path(config['path'])
        val_images_dir = dataset_path / config['val']
        val_labels_dir = dataset_path / "labels" / "val"
        
        # Get validation images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(val_images_dir.glob(f"*{ext}")))
        
        if not image_files:
            logger.warning("No validation images found!")
            return
        
        logger.info(f"Found {len(image_files)} validation images")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample random validation images
        sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Create grid visualization
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(28, 28))
        fig.suptitle(f'Validation Predictions (conf > {conf_threshold})', fontsize=16, fontweight='bold')
        
        for i, img_path in enumerate(sampled_images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run prediction
            results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
            
            # Display image
            ax.imshow(image_rgb)
            
            # Count predictions and ground truth
            num_predictions = 0
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'obb') and result.obb is not None:
                    boxes = result.obb
                    num_predictions = len(boxes)
                    
                    # Draw predicted boxes in green
                    for box in boxes:
                        if hasattr(box, 'xyxyxyxy'):
                            # OBB format - 4 corner points
                            corners = box.xyxyxyxy.cpu().numpy()
                            if len(corners) > 0:
                                points = corners[0].reshape(-1, 2)
                                polygon = Polygon(points, linewidth=2, edgecolor='lime', facecolor='none', label='Prediction')
                                ax.add_patch(polygon)
                        elif hasattr(box, 'xyxy'):
                            # Regular bbox format
                            xyxy = box.xyxy.cpu().numpy()[0]
                            x1, y1, x2, y2 = xyxy
                            rect = patches.Rectangle(
                                (x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='lime', facecolor='none'
                            )
                            ax.add_patch(rect)
            
            # Load and draw ground truth in red
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            num_gt = 0
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            num_gt += 1
                            coords = list(map(float, parts[1:]))
                            img_h, img_w = image.shape[:2]
                            
                            if len(coords) >= 8:
                                # OBB format
                                points = []
                                for j in range(0, 8, 2):
                                    x = coords[j] * img_w
                                    y = coords[j+1] * img_h
                                    points.append([x, y])
                                polygon = Polygon(points, linewidth=1, edgecolor='red', facecolor='none', linestyle='--', label='Ground Truth')
                                ax.add_patch(polygon)
                            else:
                                # Regular bbox
                                x_center, y_center, width, height = coords[:4]
                                x_center *= img_w
                                y_center *= img_h
                                width *= img_w
                                height *= img_h
                                x1 = x_center - width / 2
                                y1 = y_center - height / 2
                                rect = patches.Rectangle(
                                    (x1, y1), width, height,
                                    linewidth=1, edgecolor='red', facecolor='none', linestyle='--'
                                )
                                ax.add_patch(rect)
            
            ax.set_title(f'{img_path.name}\nGT: {num_gt} | Pred: {num_predictions}', fontsize=8)
            ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Ground Truth'),
            Patch(facecolor='none', edgecolor='lime', label='Prediction')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = output_path / f"validation_predictions_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Validation predictions saved to: {viz_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing validation predictions: {e}", exc_info=True)


def train_yolo_obb(
    dataset_config: str,
    model_name: str = "yolo11n-obb.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "",
    project: str = "trading_cards_obb",
    name: str = "yolo11n_obb_v1",
    resume: bool = False,
    pretrained: bool = True,
    save_period: int = 10,
    patience: int = 50,
    use_albumentations: bool = True,
    aug_intensity: str = "medium",
    **kwargs
) -> str:
    """
    Train YOLO OBB model with optional enhanced albumentations.
    
    Args:
        dataset_config: Path to dataset YAML configuration
        model_name: Model architecture name
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size
        device: Device to use (auto-detect if empty)
        project: Project name for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
        pretrained: Use pretrained weights
        save_period: Save checkpoint every N epochs
        patience: Early stopping patience
        use_albumentations: Whether to use enhanced albumentations (requires albumentations package)
        aug_intensity: Augmentation intensity level ("heavy", "medium", "light")
        **kwargs: Additional training arguments
    
    Returns:
        Path to best model weights
    """
    
    print("=" * 60)
    print("🚀 Starting YOLO OBB Training for Trading Card Detection")
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        print("   Enhanced with Albumentations for Orientation Robustness")
    print("=" * 60)
    
    # Load and potentially enhance configuration
    config = load_config(dataset_config)
    if use_albumentations:
        config = setup_enhanced_augmentations(config, use_albumentations)
    
    # Validate dataset
    logger.info("📊 Validating dataset...")
    if not validate_dataset(dataset_config):
        raise ValueError("Dataset validation failed!")
    
    # STEP 1: Visualize RAW dataset samples (no augmentation)
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Visualizing RAW dataset samples (no augmentation)")
    logger.info("="*60)
    raw_samples_dir = Path(project) / name / "debug_visualizations" / "01_raw_samples"
    visualize_raw_dataset_samples(
        dataset_config,
        str(raw_samples_dir),
        num_examples=16
    )
    
    # STEP 2: Generate augmentation examples if requested
    if kwargs.get('show_examples', True):
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Generating augmented training examples")
        logger.info("="*60)
        examples_dir = Path(project) / name / "debug_visualizations" / "02_augmented_examples"
        visualize_augmented_examples(
            dataset_config, 
            str(examples_dir), 
            num_examples=kwargs.get('num_examples', 8),
            use_albumentations=use_albumentations
        )
    
    # Initialize model
    print(f"\n🤖 Initializing {model_name} model...")
    if pretrained:
        print("   Using pretrained weights")
        model = YOLO(model_name)
    else:
        print("   Training from scratch")
        # For training from scratch, use the .yaml config instead of .pt weights
        model_config = model_name.replace('.pt', '.yaml')
        model = YOLO(model_config)
    
    # Setup device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    # Print training configuration
    logger.info(f"⚙️  Training Configuration:")
    logger.info(f"   Dataset: {dataset_config}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Image size: {imgsz}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Project: {project}")
    logger.info(f"   Name: {name}")
    logger.info(f"   Albumentations: {use_albumentations and ALBUMENTATIONS_AVAILABLE}")
    
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        albumentations_config = config.get('albumentations', {})
        logger.info(f"   Augmentation intensity: {albumentations_config.get('intensity', aug_intensity)}")
        logger.info(f"   Max rotation: {albumentations_config.get('max_rotation', 60)}°")
        logger.info(f"   Orientation focus: {albumentations_config.get('orientation_focus', True)}")
        
        # Log active augmentation settings
        logger.info("   Active augmentations:")
        aug_config = config.get('augmentation', {})
        key_augs = ['degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 
                   'mosaic', 'mixup', 'copy_paste', 'blur', 'noise', 'erasing']
        for key in key_augs:
            if key in aug_config:
                logger.info(f"     {key}: {aug_config[key]}")
    
    # Prepare training arguments
    training_args = {
        'data': dataset_config,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'resume': resume,
        'save_period': save_period,
        'patience': patience,
        'plots': True,
        'verbose': True,
    }
    
    # Add enhanced augmentation settings if enabled
    if use_albumentations and ALBUMENTATIONS_AVAILABLE and 'augmentation' in config:
        aug_config = config['augmentation']
        
        # Core geometric augmentations
        training_args.update({
            'degrees': aug_config.get('degrees', 60),
            'translate': aug_config.get('translate', 0.15),
            'scale': aug_config.get('scale', 0.6),
            'shear': aug_config.get('shear', 20),
            'perspective': aug_config.get('perspective', 0.0005),
        })
        
        # Flip augmentations
        training_args.update({
            'flipud': aug_config.get('flipud', 0.3),
            'fliplr': aug_config.get('fliplr', 0.5),
        })
        
        # Color and photometric augmentations
        training_args.update({
            'hsv_h': aug_config.get('hsv_h', 0.02),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'hsv_v': aug_config.get('hsv_v', 0.4),
        })
        
        # Advanced composition augmentations
        training_args.update({
            'mosaic': aug_config.get('mosaic', 1.0),
            'mixup': aug_config.get('mixup', 0.1),
            'copy_paste': aug_config.get('copy_paste', 0.15),
        })
        
        # Note: blur and noise are not valid YOLO training arguments
        # They are handled through albumentations if enabled
        
        # Lighting and exposure augmentations
        training_args.update({
            'auto_augment': aug_config.get('auto_augment', 'randaugment'),
            'erasing': aug_config.get('erasing', 0.4),
        })
        
        # Crop and resize augmentations
        training_args.update({
            'crop_fraction': aug_config.get('crop_fraction', 1.0),
        })
    
    # Add any additional kwargs, but filter out custom arguments that aren't valid YOLO args
    custom_args = {'show_examples', 'num_examples', 'validation_interval'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in custom_args}
    training_args.update(filtered_kwargs)
    
    # Setup periodic validation prediction callback
    periodic_viz_dir = Path(project) / name / "periodic_validations"
    validation_callback = create_periodic_validation_callback(
        dataset_config=dataset_config,
        output_base_dir=str(periodic_viz_dir),
        interval=kwargs.get('validation_interval', 10),  # Every 10 epochs by default
        yolo_wrapper=model  # Pass the YOLO wrapper, not trainer.model
    )
    
    # Add custom callback to YOLO
    callbacks.add_integration_callbacks(model)
    model.add_callback("on_train_epoch_end", validation_callback)
    
    # Start training
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Starting model training")
    logger.info("="*60)
    logger.info("🏋️  Training in progress...")
    logger.info("Note: YOLO will generate train_batch*.jpg files showing training batches")
    logger.info("These files are located in the training output directory")
    logger.info(f"Periodic validation predictions will be saved every {kwargs.get('validation_interval', 10)} epochs")
    logger.info(f"Location: {periodic_viz_dir}")
    logger.info("="*60 + "\n")
    
    try:
        results = model.train(**training_args)
        
        # Get best model path
        best_model_path = results.save_dir / "weights" / "best.pt"
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Best model saved to: {best_model_path}")
        
        # STEP 4: Generate validation predictions visualization
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Generating final validation predictions")
        logger.info("="*60)
        val_viz_dir = Path(project) / name / "debug_visualizations" / "03_validation_predictions"
        visualize_validation_predictions(
            model,
            dataset_config,
            str(val_viz_dir),
            num_samples=16,
            conf_threshold=0.25
        )
        
        # Log to MLflow if available
        try:
            if mlflow.active_run():
                mlflow.log_artifact(str(best_model_path))
                mlflow.log_metrics({
                    "final_mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                    "final_mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                })
                mlflow.end_run()
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
        
        return str(best_model_path)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise


def print_multi_dataset_help():
    """Print detailed help for multi-dataset training."""
    help_text = """
📊 MULTI-DATASET TRAINING GUIDE

OVERVIEW:
  Train on multiple datasets simultaneously, with control over how they're
  combined for training and evaluation. Perfect for combining synthetic and
  hand-crafted datasets.

USAGE MODES:
  1. Single dataset (default):
     python train_yolo_obb.py --data dataset.yaml
  
  2. Multiple datasets with explicit paths:
     python train_yolo_obb.py --datasets synthetic.yaml handcrafted.yaml
  
  3. Synthetic + Hand-crafted (recommended):
     python train_yolo_obb.py \\
       --synthetic-data path/to/synthetic.yaml \\
       --handcrafted-data path/to/handcrafted.yaml \\
       --eval-handcrafted-ratio 0.8 \\
       --handcrafted-aug-multiplier 3

ARGUMENTS:
  --datasets [PATHS ...]
      Multiple dataset config files to merge. Last one is treated as hand-crafted.
  
  --synthetic-data PATH
      Path to synthetic dataset config (use with --handcrafted-data)
  
  --handcrafted-data PATH
      Path to hand-crafted dataset config (use with --synthetic-data)
  
  --eval-handcrafted-ratio FLOAT
      Ratio of hand-crafted data in validation set (0.0-1.0)
      Default: 0.8 (80% hand-crafted, 20% synthetic in validation)
      Higher values = more focus on real-world performance
  
  --handcrafted-aug-multiplier INT
      How many times to duplicate hand-crafted training data
      Default: 3 (triples the hand-crafted training examples)
      For small datasets (300 images vs 60K synthetic), try 20-100x
      This stretches the hand-crafted dataset through augmentation
  
  --apply-pre-augmentation
      Apply deterministic pre-augmentations to each duplicate
      Without this: duplicates are identical (get different random augs during training)
      With this: each duplicate gets unique pre-applied PHOTOMETRIC augmentations
                 (brightness, contrast, color, blur, noise - NO geometric transforms)
      Note: Rotation/scaling handled by YOLO to preserve bounding box accuracy
      Recommended for high multipliers (20x+) to increase diversity
  
  --merged-dataset-dir PATH
      Directory to save the merged dataset
      Default: ../data/merged_dataset

HOW IT WORKS:
  
  Training Set:
    - All synthetic images (1x)
    - All hand-crafted images (multiplied by --handcrafted-aug-multiplier)
    - Each duplicate gets different augmentations during training
    - This balances the dataset and stretches hand-crafted data
  
  Validation Set:
    - Sampled according to --eval-handcrafted-ratio
    - Higher ratio = model evaluated more on hand-crafted (real) data
    - Ensures metrics reflect real-world performance
  
  Augmentation:
    - Same augmentation settings apply to all training data
    - Hand-crafted duplicates receive different random augmentations each epoch
    - Configure augmentation intensity with --aug-intensity or individual flags

EXAMPLES:
  
  # Basic: 80% hand-crafted validation, 3x hand-crafted training
  python train_yolo_obb.py \\
    --synthetic-data data/synthetic.yaml \\
    --handcrafted-data data/handcrafted.yaml
  
  # Heavy hand-crafted focus: 90% validation, 5x training multiplier
  python train_yolo_obb.py \\
    --synthetic-data data/synthetic.yaml \\
    --handcrafted-data data/handcrafted.yaml \\
    --eval-handcrafted-ratio 0.9 \\
    --handcrafted-aug-multiplier 5 \\
    --aug-intensity heavy
  
  # Balanced approach: 70% validation, 2x multiplier
  python train_yolo_obb.py \\
    --synthetic-data data/synthetic.yaml \\
    --handcrafted-data data/handcrafted.yaml \\
    --eval-handcrafted-ratio 0.7 \\
    --handcrafted-aug-multiplier 2
  
  # Multiple datasets (3 datasets, last is hand-crafted)
  python train_yolo_obb.py \\
    --datasets synthetic1.yaml synthetic2.yaml handcrafted.yaml \\
    --eval-handcrafted-ratio 0.75
  
  # EXTREME stretching for tiny hand-crafted dataset (300 vs 60K images)
  # Option A: 50x multiplier with pre-augmentation (15,000 effective images)
  python train_yolo_obb.py \\
    --synthetic-data data/synthetic.yaml \\
    --handcrafted-data data/handcrafted.yaml \\
    --handcrafted-aug-multiplier 50 \\
    --apply-pre-augmentation \\
    --eval-handcrafted-ratio 0.9
  
  # Option B: 100x multiplier with pre-augmentation (30,000 effective images)
  python train_yolo_obb.py \\
    --synthetic-data data/synthetic.yaml \\
    --handcrafted-data data/handcrafted.yaml \\
    --handcrafted-aug-multiplier 100 \\
    --apply-pre-augmentation \\
    --eval-handcrafted-ratio 0.9 \\
    --aug-intensity heavy

TIPS:
  - Higher multiplier = more training epochs before overfitting hand-crafted data
  - Higher eval ratio = metrics better reflect real-world performance
  - Use --apply-pre-augmentation with high multipliers (20x+) for max diversity
  - For 300 vs 60K imbalance, try 50x multiplier (15K effective images)
  - For 300 vs 60K imbalance, try 100x multiplier (30K effective images)
  - Start with defaults (0.8 ratio, 3x multiplier) and adjust based on results
  - Monitor both training and validation metrics to detect overfitting
  - Use --show-examples to visualize merged dataset augmentations

UNDERSTANDING THE MATH:
  If you have 300 hand-crafted and 60,000 synthetic images:
  - 3x multiplier: 900 hand-crafted (1.5% of training data)
  - 10x multiplier: 3,000 hand-crafted (5% of training data)
  - 20x multiplier: 6,000 hand-crafted (9% of training data)
  - 50x multiplier: 15,000 hand-crafted (20% of training data) ← RECOMMENDED
  - 100x multiplier: 30,000 hand-crafted (33% of training data)
  - 200x multiplier: 60,000 hand-crafted (50% of training data)
  
  YOLO's augmentation pipeline will apply additional random augmentations during
  training on top of any pre-augmentations, creating even more diversity.
"""
    print(help_text)


def print_augmentation_help():
    """Print detailed help for augmentation options."""
    help_text = """
🎨 AUGMENTATION OPTIONS GUIDE

GEOMETRIC AUGMENTATIONS:
  --degrees FLOAT        Rotation range (±degrees). Default: 60.0
                        Higher values = more orientation robustness
  --translate FLOAT      Translation range (±fraction). Default: 0.15
                        Simulates camera position variations
  --scale FLOAT          Scale variation (±gain). Default: 0.6
                        Handles different card sizes and distances
  --shear FLOAT          Shear transformation (±degrees). Default: 20.0
                        Simulates perspective viewing angles
  --perspective FLOAT    Perspective distortion (±fraction). Default: 0.0005
                        Adds 3D viewing angle variations

FLIP AUGMENTATIONS:
  --flipud FLOAT         Vertical flip probability. Default: 0.3
                        Useful for cards that can be upside down
  --fliplr FLOAT         Horizontal flip probability. Default: 0.5
                        Mirrors card orientation

COLOR & LIGHTING:
  --hsv-h FLOAT          Hue shift (fraction). Default: 0.02
                        Color temperature variations
  --hsv-s FLOAT          Saturation shift (fraction). Default: 0.7
                        Color intensity variations
  --hsv-v FLOAT          Value/brightness shift (fraction). Default: 0.4
                        Lighting condition variations

ADVANCED COMPOSITION:
  --mosaic FLOAT         Mosaic augmentation probability. Default: 1.0
                        Combines multiple images for context
  --mixup FLOAT          MixUp augmentation probability. Default: 0.1
                        Blends images for regularization
  --copy-paste FLOAT     Copy-paste probability. Default: 0.15
                        Increases object diversity

QUALITY & ROBUSTNESS:
  --blur FLOAT           Motion blur probability. Default: 0.01
                        Simulates camera shake
  --noise FLOAT          Gaussian noise probability. Default: 0.02
                        Simulates sensor noise
  --erasing FLOAT        Random erasing probability. Default: 0.4
                        Cutout-style occlusion simulation

AUTO AUGMENTATION:
  --auto-augment STR     Policy: randaugment/autoaugment/augmix
                        Automatic augmentation strategies

INTENSITY PRESETS:
  --aug-intensity STR    heavy/medium/light. Default: medium
                        Predefined augmentation combinations

VISUALIZATION OPTIONS:
  --show-examples        Generate augmented examples during training
  --num-examples INT     Number of examples to generate (default: 8)
  --examples-only        Only generate examples, don't train

EXAMPLES:
  # Maximum orientation robustness
  python train_yolo_obb.py --degrees 90 --shear 30 --flipud 0.5

  # Light augmentation for fine-tuning
  python train_yolo_obb.py --aug-intensity light --degrees 15

  # Heavy augmentation for initial training
  python train_yolo_obb.py --aug-intensity heavy --mixup 0.2

  # Generate augmentation examples only
  python train_yolo_obb.py --examples-only --num-examples 12

  # Train with example generation
  python train_yolo_obb.py --show-examples --aug-intensity medium
"""
    print(help_text)


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Train YOLO OBB model for trading card detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --aug-help for augmentation guide | Use --multi-dataset-help for multi-dataset training guide"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data", 
        type=str, 
        default="../configs/yolo_obb_dataset.yaml",
        help="Path to dataset YAML configuration file (single dataset mode)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        help="Multiple dataset configs to merge (e.g., --datasets synthetic.yaml handcrafted.yaml)"
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        help="Path to synthetic dataset config (alternative to --datasets)"
    )
    parser.add_argument(
        "--handcrafted-data",
        type=str,
        help="Path to hand-crafted dataset config (alternative to --datasets)"
    )
    parser.add_argument(
        "--eval-handcrafted-ratio",
        type=float,
        default=0.8,
        help="Ratio of hand-crafted data in validation set (0.0-1.0). Default: 0.8"
    )
    parser.add_argument(
        "--handcrafted-aug-multiplier",
        type=int,
        default=3,
        help="How many times to duplicate hand-crafted training data. Default: 3. For small datasets (300 images), try 20-100x"
    )
    parser.add_argument(
        "--apply-pre-augmentation",
        action="store_true",
        help="Apply deterministic pre-augmentations to hand-crafted duplicates (rotation, brightness, contrast, etc). Increases diversity."
    )
    parser.add_argument(
        "--merged-dataset-dir",
        type=str,
        default="../data/merged_dataset",
        help="Directory to save merged dataset. Default: ../data/merged_dataset"
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device (auto-detect if empty)")
    
    # Training arguments
    parser.add_argument("--project", type=str, default="trading_cards_obb", help="Project name")
    parser.add_argument("--name", type=str, default="yolo11n_obb_v1", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--fraction", type=float, help="Fraction of validation set to use (0.0-1.0). Reduces GPU memory usage.")
    
    # Augmentation arguments
    parser.add_argument("--no-albumentations", action="store_true", help="Disable enhanced albumentations")
    parser.add_argument("--aug-intensity", type=str, default="medium", 
                       choices=["heavy", "medium", "light"], help="Augmentation intensity level")
    
    # Geometric augmentation controls
    parser.add_argument("--degrees", type=float, help="Image rotation range (+/- degrees)")
    parser.add_argument("--translate", type=float, help="Image translation (+/- fraction)")
    parser.add_argument("--scale", type=float, help="Image scale (+/- gain)")
    parser.add_argument("--shear", type=float, help="Image shear (+/- degrees)")
    parser.add_argument("--perspective", type=float, help="Image perspective (+/- fraction)")
    
    # Flip augmentation controls
    parser.add_argument("--flipud", type=float, help="Image flip up-down probability")
    parser.add_argument("--fliplr", type=float, help="Image flip left-right probability")
    
    # Color augmentation controls
    parser.add_argument("--hsv-h", type=float, help="HSV-Hue augmentation (fraction)")
    parser.add_argument("--hsv-s", type=float, help="HSV-Saturation augmentation (fraction)")
    parser.add_argument("--hsv-v", type=float, help="HSV-Value augmentation (fraction)")
    
    # Advanced augmentation controls
    parser.add_argument("--mosaic", type=float, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, help="MixUp augmentation probability")
    parser.add_argument("--copy-paste", type=float, help="Copy-paste augmentation probability")
    parser.add_argument("--blur", type=float, help="Motion blur probability")
    parser.add_argument("--noise", type=float, help="Gaussian noise probability")
    parser.add_argument("--erasing", type=float, help="Random erasing probability")
    parser.add_argument("--auto-augment", type=str, choices=["randaugment", "autoaugment", "augmix"], 
                       help="Auto augmentation policy")
    parser.add_argument("--crop-fraction", type=float, help="Crop fraction for training")
    
    # MLflow arguments
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    
    # Visualization arguments
    parser.add_argument("--show-examples", action="store_true", help="Generate and save augmented training examples")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of augmentation examples to generate")
    parser.add_argument("--examples-only", action="store_true", help="Only generate examples, don't train")
    parser.add_argument("--validation-interval", type=int, default=10, help="Generate validation predictions every N epochs (default: 10)")
    
    # Help arguments
    parser.add_argument("--aug-help", action="store_true", help="Show detailed augmentation options guide")
    parser.add_argument("--multi-dataset-help", action="store_true", help="Show detailed multi-dataset training guide")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.multi_dataset_help:
        print_multi_dataset_help()
        return
    
    if args.aug_help:
        print_augmentation_help()
        return
    
    # Determine which dataset mode to use
    dataset_configs = []
    
    # Priority: --datasets > (--synthetic-data + --handcrafted-data) > --data
    if args.datasets:
        dataset_configs = args.datasets
        logger.info(f"Multi-dataset mode: Using {len(dataset_configs)} datasets")
    elif args.synthetic_data and args.handcrafted_data:
        # Synthetic first, hand-crafted last (so it gets index -1)
        dataset_configs = [args.synthetic_data, args.handcrafted_data]
        logger.info("Multi-dataset mode: Using synthetic + hand-crafted datasets")
    elif args.synthetic_data or args.handcrafted_data:
        print("Error: Both --synthetic-data and --handcrafted-data must be specified together")
        sys.exit(1)
    
    # Merge datasets if multiple are specified
    if len(dataset_configs) >= 2:
        logger.info("\n" + "="*60)
        logger.info("MERGING MULTIPLE DATASETS")
        logger.info("="*60)
        logger.info(f"Hand-crafted augmentation multiplier: {args.handcrafted_aug_multiplier}x")
        logger.info(f"Evaluation hand-crafted ratio: {args.eval_handcrafted_ratio:.1%}")
        logger.info("="*60 + "\n")
        
        # Merge datasets
        merged_config_path = merge_datasets(
            dataset_configs=dataset_configs,
            output_dir=args.merged_dataset_dir,
            eval_handcrafted_ratio=args.eval_handcrafted_ratio,
            handcrafted_aug_multiplier=args.handcrafted_aug_multiplier,
            handcrafted_dataset_index=-1,  # Last dataset is hand-crafted
            apply_pre_augmentation=args.apply_pre_augmentation,
        )
        
        # Use merged dataset for training
        args.data = merged_config_path
        logger.info(f"Using merged dataset config: {merged_config_path}\n")
    
    # Generate examples only if requested
    if args.examples_only:
        print("🎨 Generating augmentation examples only...")
        examples_dir = f"{args.project}/{args.name}/augmentation_examples"
        visualize_augmented_examples(
            args.data, 
            examples_dir, 
            num_examples=args.num_examples,
            use_albumentations=not args.no_albumentations
        )
        print("✅ Examples generation complete!")
        return
    
    # Load configuration if available
    config = {}
    if os.path.exists(args.data):
        try:
            config = load_config(args.data)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Setup MLflow
    if not args.no_mlflow and config:
        try:
            setup_mlflow(config)
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")
    
    # Override config with command line arguments
    training_args = {
        "dataset_config": args.data,
        "model_name": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "pretrained": not args.no_pretrained,
        "save_period": args.save_period,
        "patience": args.patience,
        "use_albumentations": not args.no_albumentations,
        "aug_intensity": args.aug_intensity,
    }
    
    # Add fraction if specified (for reducing validation memory usage)
    if args.fraction is not None:
        training_args["fraction"] = args.fraction
    
    # Add command line augmentation overrides
    augmentation_overrides = {}
    
    # Geometric augmentations
    if args.degrees is not None:
        augmentation_overrides['degrees'] = args.degrees
    if args.translate is not None:
        augmentation_overrides['translate'] = args.translate
    if args.scale is not None:
        augmentation_overrides['scale'] = args.scale
    if args.shear is not None:
        augmentation_overrides['shear'] = args.shear
    if args.perspective is not None:
        augmentation_overrides['perspective'] = args.perspective
    
    # Flip augmentations
    if args.flipud is not None:
        augmentation_overrides['flipud'] = args.flipud
    if args.fliplr is not None:
        augmentation_overrides['fliplr'] = args.fliplr
    
    # Color augmentations
    if getattr(args, 'hsv_h', None) is not None:
        augmentation_overrides['hsv_h'] = args.hsv_h
    if getattr(args, 'hsv_s', None) is not None:
        augmentation_overrides['hsv_s'] = args.hsv_s
    if getattr(args, 'hsv_v', None) is not None:
        augmentation_overrides['hsv_v'] = args.hsv_v
    
    # Advanced augmentations
    if args.mosaic is not None:
        augmentation_overrides['mosaic'] = args.mosaic
    if args.mixup is not None:
        augmentation_overrides['mixup'] = args.mixup
    if getattr(args, 'copy_paste', None) is not None:
        augmentation_overrides['copy_paste'] = args.copy_paste
    if args.blur is not None:
        augmentation_overrides['blur'] = args.blur
    if args.noise is not None:
        augmentation_overrides['noise'] = args.noise
    if args.erasing is not None:
        augmentation_overrides['erasing'] = args.erasing
    if getattr(args, 'auto_augment', None) is not None:
        augmentation_overrides['auto_augment'] = args.auto_augment
    if getattr(args, 'crop_fraction', None) is not None:
        augmentation_overrides['crop_fraction'] = args.crop_fraction
    
    # Add augmentation overrides to training args
    training_args.update(augmentation_overrides)
    
    # Add visualization arguments
    training_args.update({
        'show_examples': args.show_examples,
        'num_examples': args.num_examples,
        'validation_interval': args.validation_interval,
    })
    
    # Start training
    try:
        best_model_path = train_yolo_obb(**training_args)
        print(f"\n🎉 Training completed! Best model: {best_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
