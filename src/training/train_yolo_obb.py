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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
    **kwargs
) -> str:
    """
    Train YOLO OBB model.
    
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
        **kwargs: Additional training arguments
    
    Returns:
        Path to best model weights
    """
    
    print("=" * 60)
    print("🚀 Starting YOLO OBB Training for Trading Card Detection")
    print("=" * 60)
    
    # Validate dataset
    print("\n📊 Validating dataset...")
    if not validate_dataset(dataset_config):
        raise ValueError("Dataset validation failed!")
    
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
    print(f"\n⚙️  Training Configuration:")
    print(f"   Dataset: {dataset_config}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Project: {project}")
    print(f"   Name: {name}")
    
    # Start training
    print(f"\n🏋️  Starting training...")
    try:
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            resume=resume,
            save_period=save_period,
            patience=patience,
            plots=True,
            verbose=True,
            **kwargs
        )
        
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


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description="Train YOLO OBB model for trading card detection")
    
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
    
    # MLflow arguments
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    
    args = parser.parse_args()
    
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
    }
    
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
