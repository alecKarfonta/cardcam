#!/usr/bin/env python3
"""
Custom dataset class with integrated albumentations augmentation pipeline.
Designed for YOLO OBB training with enhanced orientation variations.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from .albumentations_augmentations import OrientationAugmentations, YOLOOBBAugmentations

logger = logging.getLogger(__name__)


class AugmentedCardDataset(Dataset):
    """
    Custom dataset class with albumentations integration for card detection.
    Supports both COCO and YOLO annotation formats with enhanced augmentations.
    """
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 image_size: Tuple[int, int] = (640, 640),
                 augmentation_intensity: str = 'medium',
                 annotation_format: str = 'yolo',
                 use_albumentations: bool = True,
                 cache_images: bool = False):
        """
        Initialize augmented card dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (width, height)
            augmentation_intensity: 'heavy', 'medium', 'light', or 'none'
            annotation_format: 'yolo' or 'coco'
            use_albumentations: Whether to use albumentations pipeline
            cache_images: Whether to cache images in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augmentation_intensity = augmentation_intensity
        self.annotation_format = annotation_format
        self.use_albumentations = use_albumentations
        self.cache_images = cache_images
        
        # Setup paths
        self.images_dir = self.data_dir / "images" / split
        self.labels_dir = self.data_dir / "labels" / split
        
        # Load image and annotation paths
        self.image_paths = self._load_image_paths()
        self.annotation_paths = self._load_annotation_paths()
        
        # Filter valid pairs
        self._filter_valid_pairs()
        
        # Setup augmentation pipeline
        self._setup_augmentations()
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.image_paths)} samples")
        logger.info(f"Split: {split}, Format: {annotation_format}, Augmentation: {augmentation_intensity}")
    
    def _load_image_paths(self) -> List[Path]:
        """Load all image file paths."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(self.images_dir.glob(f"*{ext}")))
            image_paths.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        return sorted(image_paths)
    
    def _load_annotation_paths(self) -> List[Path]:
        """Load annotation file paths based on format."""
        if self.annotation_format == 'yolo':
            return [self.labels_dir / f"{img_path.stem}.txt" for img_path in self.image_paths]
        elif self.annotation_format == 'coco':
            # Look for COCO annotation file
            coco_file = self.data_dir / "annotations" / f"{self.split}_annotations.json"
            if coco_file.exists():
                return [coco_file] * len(self.image_paths)
            else:
                raise FileNotFoundError(f"COCO annotation file not found: {coco_file}")
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
    
    def _filter_valid_pairs(self):
        """Filter image-annotation pairs to keep only valid ones."""
        valid_pairs = []
        valid_annotations = []
        
        for img_path, ann_path in zip(self.image_paths, self.annotation_paths):
            if self.annotation_format == 'yolo' and ann_path.exists():
                valid_pairs.append(img_path)
                valid_annotations.append(ann_path)
            elif self.annotation_format == 'coco' and ann_path.exists():
                valid_pairs.append(img_path)
                valid_annotations.append(ann_path)
        
        self.image_paths = valid_pairs
        self.annotation_paths = valid_annotations
        
        logger.info(f"Found {len(self.image_paths)} valid image-annotation pairs")
    
    def _setup_augmentations(self):
        """Setup augmentation pipelines."""
        if not self.use_albumentations or self.augmentation_intensity == 'none':
            # Minimal pipeline for validation or no augmentation
            self.transform = A.Compose([
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo' if self.annotation_format == 'yolo' else 'coco',
                label_fields=['class_labels']
            ))
        else:
            # Use orientation-focused augmentations
            if self.split == 'train':
                # Training augmentations
                orientation_aug = OrientationAugmentations()
                base_pipeline = orientation_aug.get_pipeline(self.augmentation_intensity)
                
                # Add resize and normalization
                self.transform = A.Compose([
                    A.Resize(height=self.image_size[1], width=self.image_size[0]),
                    base_pipeline,
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(
                    format='yolo' if self.annotation_format == 'yolo' else 'coco',
                    label_fields=['class_labels'],
                    min_visibility=0.3
                ))
            else:
                # Validation/test - minimal augmentation
                self.transform = A.Compose([
                    A.Resize(height=self.image_size[1], width=self.image_size[0]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(
                    format='yolo' if self.annotation_format == 'yolo' else 'coco',
                    label_fields=['class_labels']
                ))
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file or cache."""
        if self.image_cache is not None and str(image_path) in self.image_cache:
            return self.image_cache[str(image_path)]
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.image_cache is not None:
            self.image_cache[str(image_path)] = image
        
        return image
    
    def _load_yolo_annotations(self, annotation_path: Path, image_shape: Tuple[int, int]) -> Tuple[List, List]:
        """Load YOLO format annotations."""
        bboxes = []
        class_labels = []
        
        if not annotation_path.exists():
            return bboxes, class_labels
        
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # YOLO format: class_id, x_center, y_center, width, height (normalized)
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def _load_coco_annotations(self, image_path: Path) -> Tuple[List, List]:
        """Load COCO format annotations for specific image."""
        # This is a simplified version - in practice, you'd load the full COCO file once
        # and index by image filename
        bboxes = []
        class_labels = []
        
        # For now, return empty annotations
        # TODO: Implement full COCO annotation loading
        return bboxes, class_labels
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with augmentations applied."""
        # Load image
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        
        # Load annotations
        if self.annotation_format == 'yolo':
            bboxes, class_labels = self._load_yolo_annotations(
                self.annotation_paths[idx], 
                image.shape[:2]
            )
        else:
            bboxes, class_labels = self._load_coco_annotations(image_path)
        
        # Apply augmentations
        try:
            if bboxes:
                augmented = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                image = augmented['image']
                bboxes = augmented['bboxes']
                class_labels = augmented['class_labels']
            else:
                # No bboxes - apply image-only transforms
                augmented = A.Compose([
                    A.Resize(height=self.image_size[1], width=self.image_size[0]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])(image=image)
                image = augmented['image']
        
        except Exception as e:
            logger.warning(f"Augmentation failed for {image_path}: {e}")
            # Fallback to simple resize and normalize
            simple_transform = A.Compose([
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            image = simple_transform(image=image)['image']
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4)),
            'class_labels': torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.empty((0,)),
            'image_path': str(image_path),
            'image_id': idx
        }
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about augmentation pipeline."""
        return {
            'dataset_size': len(self),
            'split': self.split,
            'image_size': self.image_size,
            'augmentation_intensity': self.augmentation_intensity,
            'annotation_format': self.annotation_format,
            'use_albumentations': self.use_albumentations,
            'cached_images': len(self.image_cache) if self.image_cache else 0
        }


class YOLOOBBDataset(Dataset):
    """
    Specialized dataset for YOLO OBB training with albumentations.
    Optimized for oriented bounding box detection.
    """
    
    def __init__(self,
                 config_path: Union[str, Path],
                 split: str = 'train',
                 image_size: Tuple[int, int] = (640, 640),
                 use_augmentations: bool = True):
        """
        Initialize YOLO OBB dataset.
        
        Args:
            config_path: Path to YOLO dataset configuration file
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (width, height)
            use_augmentations: Whether to apply augmentations
        """
        self.config_path = Path(config_path)
        self.split = split
        self.image_size = image_size
        self.use_augmentations = use_augmentations
        
        # Load dataset configuration
        self.config = self._load_config()
        
        # Setup paths
        self.data_root = Path(self.config['path'])
        self.images_dir = self.data_root / self.config[split]
        self.labels_dir = self.data_root / "labels" / split
        
        # Load samples
        self.samples = self._load_samples()
        
        # Setup augmentations
        self.yolo_aug = YOLOOBBAugmentations(image_size=image_size)
        
        logger.info(f"Initialized YOLO OBB dataset with {len(self.samples)} samples")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YOLO dataset configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """Load image-label pairs."""
        samples = []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        for ext in image_extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    samples.append((img_path, label_path))
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels (YOLO OBB format)
        bboxes, class_labels = self._load_obb_labels(label_path)
        
        # Apply augmentations
        if self.use_augmentations and self.split == 'train':
            pipeline = self.yolo_aug.get_train_pipeline()
        else:
            pipeline = self.yolo_aug.get_val_pipeline()
        
        try:
            if bboxes:
                augmented = pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = augmented['image']
                bboxes = augmented['bboxes']
                class_labels = augmented['class_labels']
            else:
                # No annotations - image only
                image = pipeline(image=image)['image']
        except Exception as e:
            logger.warning(f"Augmentation failed for {img_path}: {e}")
            # Fallback
            fallback = A.Compose([
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            image = fallback(image=image)['image']
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 5)),
            'class_labels': torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.empty((0,)),
            'image_path': str(img_path)
        }
    
    def _load_obb_labels(self, label_path: Path) -> Tuple[List, List]:
        """Load YOLO OBB format labels."""
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:  # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:9]))
                    
                    # Convert to center format for albumentations
                    # This is simplified - you may need to adjust based on your OBB format
                    x_coords = coords[::2]
                    y_coords = coords[1::2]
                    x_center = sum(x_coords) / 4
                    y_center = sum(y_coords) / 4
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels


def create_augmented_dataloader(data_dir: str,
                              split: str = 'train',
                              batch_size: int = 16,
                              num_workers: int = 4,
                              image_size: Tuple[int, int] = (640, 640),
                              augmentation_intensity: str = 'medium') -> torch.utils.data.DataLoader:
    """
    Create DataLoader with augmented dataset.
    
    Args:
        data_dir: Path to dataset directory
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation_intensity: Augmentation intensity level
        
    Returns:
        DataLoader instance
    """
    dataset = AugmentedCardDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        augmentation_intensity=augmentation_intensity,
        use_albumentations=True
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )


if __name__ == "__main__":
    # Test the dataset
    dataset = AugmentedCardDataset(
        data_dir="data/yolo_obb",
        split='train',
        augmentation_intensity='medium'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Augmentation stats: {dataset.get_augmentation_stats()}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Bboxes shape: {sample['bboxes'].shape}")
