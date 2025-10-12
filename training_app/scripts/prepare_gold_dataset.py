#!/usr/bin/env python3
"""
Prepare Gold Dataset for Training

Splits the gold dataset into train/val sets and creates a YAML config.
"""

import os
import shutil
from pathlib import Path
import random
import yaml

def prepare_gold_dataset(
    gold_data_dir: str = "/home/alec/git/pokemon/training_app/data/gold",
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Split gold dataset into train/val sets.
    
    Args:
        gold_data_dir: Path to gold data directory
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    gold_path = Path(gold_data_dir)
    images_dir = gold_path / "images"
    source_labels_dir = gold_path / "labels" / "test"
    
    # Create train/val directories
    train_images_dir = gold_path / "images_split" / "train"
    val_images_dir = gold_path / "images_split" / "val"
    train_labels_dir = gold_path / "labels" / "train"
    val_labels_dir = gold_path / "labels" / "val"
    
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all images that have corresponding labels
    label_files = list(source_labels_dir.glob("*.txt"))
    image_stems = [lf.stem for lf in label_files]
    
    # Find corresponding images
    valid_pairs = []
    for stem in image_stems:
        # Try different image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                label_path = source_labels_dir / f"{stem}.txt"
                valid_pairs.append((img_path, label_path))
                break
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"Train: {len(train_pairs)} images")
    print(f"Val: {len(val_pairs)} images")
    
    # Copy files to train directory
    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        shutil.copy2(label_path, train_labels_dir / label_path.name)
    
    # Copy files to val directory
    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        shutil.copy2(label_path, val_labels_dir / label_path.name)
    
    # Create dataset config
    config = {
        'path': str(gold_path.absolute()),
        'train': 'images_split/train',
        'val': 'images_split/val',
        'names': {0: 'card'},
        'nc': 1,
        'info': {
            'description': 'Gold Hand-Crafted Dataset for Trading Card Detection',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Hand-crafted gold annotations',
            'train_images': len(train_pairs),
            'val_images': len(val_pairs),
        }
    }
    
    config_path = gold_path / "gold_dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nDataset config created: {config_path}")
    print(f"\nDirectory structure:")
    print(f"  {train_images_dir}: {len(list(train_images_dir.glob('*')))} images")
    print(f"  {val_images_dir}: {len(list(val_images_dir.glob('*')))} images")
    print(f"  {train_labels_dir}: {len(list(train_labels_dir.glob('*.txt')))} labels")
    print(f"  {val_labels_dir}: {len(list(val_labels_dir.glob('*.txt')))} labels")
    
    return str(config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare gold dataset for training")
    parser.add_argument("--gold-dir", type=str, 
                       default="/home/alec/git/pokemon/training_app/data/gold",
                       help="Path to gold data directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    config_path = prepare_gold_dataset(
        gold_data_dir=args.gold_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"\nGold dataset ready for training!")
    print(f"Use config: {config_path}")

