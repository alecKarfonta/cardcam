#!/usr/bin/env python3
"""
YOLO OBB Training Script for Trading Card Detection

This script trains a YOLOv11 model with Oriented Bounding Boxes (OBB) for detecting
trading cards with arbitrary orientations.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    
    # Generate augmentation examples if requested
    if kwargs.get('show_examples', True):
        examples_dir = Path(project) / name / "augmentation_examples"
        logger.info(f"🎨 Generating augmentation examples...")
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
        
        # Quality and noise augmentations
        training_args.update({
            'blur': aug_config.get('blur', 0.01),
            'noise': aug_config.get('noise', 0.02),
        })
        
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
    custom_args = {'show_examples', 'num_examples'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in custom_args}
    training_args.update(filtered_kwargs)
    
    # Start training
    logger.info("🏋️  Starting training...")
    try:
        results = model.train(**training_args)
        
        # Get best model path
        best_model_path = results.save_dir / "weights" / "best.pt"
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Best model saved to: {best_model_path}")
        
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
        epilog="Use --aug-help for detailed augmentation options guide"
    )
    
    # Required arguments
    parser.add_argument(
        "--data", 
        type=str, 
        default="../configs/yolo_obb_dataset.yaml",
        help="Path to dataset YAML configuration file"
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
    
    # Help arguments
    parser.add_argument("--aug-help", action="store_true", help="Show detailed augmentation options guide")
    
    args = parser.parse_args()
    
    # Show augmentation help if requested
    if args.aug_help:
        print_augmentation_help()
        return
    
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
