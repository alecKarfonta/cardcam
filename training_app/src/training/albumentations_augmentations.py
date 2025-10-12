#!/usr/bin/env python3
"""
Albumentations-based augmentation pipeline for trading card detection.
Focuses on orientation variations to improve model robustness.
"""

import albumentations as A
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import random


class OrientationAugmentations:
    """
    Comprehensive orientation-focused augmentation pipeline using albumentations.
    Designed to help models handle various card orientations and rotations.
    """
    
    def __init__(self, 
                 rotation_limit: int = 45,
                 perspective_scale: float = 0.1,
                 shear_limit: int = 20,
                 brightness_contrast_prob: float = 0.7,
                 geometric_prob: float = 0.8):
        """
        Initialize orientation augmentation pipeline.
        
        Args:
            rotation_limit: Maximum rotation angle in degrees (±)
            perspective_scale: Scale for perspective transformation (0.0-1.0)
            shear_limit: Maximum shear angle in degrees (±)
            brightness_contrast_prob: Probability of applying brightness/contrast
            geometric_prob: Probability of applying geometric transformations
        """
        self.rotation_limit = rotation_limit
        self.perspective_scale = perspective_scale
        self.shear_limit = shear_limit
        self.brightness_contrast_prob = brightness_contrast_prob
        self.geometric_prob = geometric_prob
        
        # Create different augmentation pipelines for different training phases
        self.heavy_augmentation = self._create_heavy_pipeline()
        self.medium_augmentation = self._create_medium_pipeline()
        self.light_augmentation = self._create_light_pipeline()
        
    def _create_heavy_pipeline(self) -> A.Compose:
        """Create heavy augmentation pipeline for early training."""
        return A.Compose([
            # Geometric transformations - focus on orientation
            A.OneOf([
                A.Rotate(limit=self.rotation_limit, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.SafeRotate(limit=self.rotation_limit, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            ], p=self.geometric_prob),
            
            # Perspective and affine transformations
            A.OneOf([
                A.Perspective(scale=(0.02, self.perspective_scale), p=1.0),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-15, 15),
                    shear=(-self.shear_limit, self.shear_limit),
                    p=1.0,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0
                ),
            ], p=0.6),
            
            # Flip transformations for orientation variety
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Transpose(p=1.0),  # Swap width and height (90° + flip)
            ], p=0.4),
            
            # Photometric augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=self.brightness_contrast_prob),
            
            # Color augmentations
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.ChannelShuffle(p=1.0),
            ], p=0.5),
            
            # Noise and blur for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            
            # Distortions that might occur in real scenarios
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ], p=0.2),
            
            # Advanced lighting and shadow effects
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=1.0),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, p=1.0),
            ], p=0.15),
            
            # Texture and surface effects
            A.OneOf([
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Superpixels(p_replace=(0.1, 0.3), n_segments=(64, 128), p=1.0),
            ], p=0.1),
            
            # Advanced color manipulations
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.FancyPCA(alpha=0.1, p=1.0),
                A.ToSepia(p=1.0),
                A.ToGray(p=1.0),
            ], p=0.2),
            
            # Cutout and erasing effects
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
                A.GridDropout(ratio=0.5, unit_size_min=2, unit_size_max=20, holes_number_x=5, holes_number_y=5, p=1.0),
            ], p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _create_medium_pipeline(self) -> A.Compose:
        """Create medium augmentation pipeline for mid training."""
        return A.Compose([
            # Focus on rotation and basic geometric transforms
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=1.0),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-10, 10),
                    shear=(-10, 10),
                    p=1.0,
                    mode=cv2.BORDER_CONSTANT,
                    cval=0
                ),
            ], p=0.4),
            
            # Reduced flip probability
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.3),
            
            # Moderate photometric changes
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.6
            ),
            
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.4
            ),
            
            # Light noise
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_visibility=0.4
        ))
    
    def _create_light_pipeline(self) -> A.Compose:
        """Create light augmentation pipeline for fine-tuning."""
        return A.Compose([
            # Minimal rotation for fine-tuning
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Light perspective changes
            A.Perspective(scale=(0.01, 0.03), p=0.3),
            
            # Horizontal flip only
            A.HorizontalFlip(p=0.2),
            
            # Subtle brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.4
            ),
            
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_visibility=0.5
        ))
    
    def get_pipeline(self, intensity: str = "medium") -> A.Compose:
        """
        Get augmentation pipeline by intensity level.
        
        Args:
            intensity: "heavy", "medium", or "light"
            
        Returns:
            Albumentations compose pipeline
        """
        if intensity == "heavy":
            return self.heavy_augmentation
        elif intensity == "medium":
            return self.medium_augmentation
        elif intensity == "light":
            return self.light_augmentation
        else:
            raise ValueError(f"Unknown intensity: {intensity}. Use 'heavy', 'medium', or 'light'")


class YOLOOBBAugmentations:
    """
    Specialized augmentations for YOLO OBB (Oriented Bounding Box) training.
    Handles rotated bounding boxes correctly during augmentation.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (640, 640)):
        """
        Initialize YOLO OBB augmentations.
        
        Args:
            image_size: Target image size (width, height)
        """
        self.image_size = image_size
        
        # Create training pipeline optimized for OBB
        self.train_pipeline = A.Compose([
            # Resize to training size first
            A.Resize(height=image_size[1], width=image_size[0], p=1.0),
            
            # Orientation-focused augmentations
            A.OneOf([
                A.Rotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.SafeRotate(limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            ], p=0.8),
            
            # Perspective transformation
            A.Perspective(scale=(0.02, 0.08), p=0.4),
            
            # Affine transformations
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                shear=(-15, 15),
                p=0.5,
                mode=cv2.BORDER_CONSTANT,
                cval=0
            ),
            
            # Flip augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            
            # Photometric augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),
            
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.5
            ),
            
            # Noise for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
            ], p=0.3),
            
            # Blur augmentations
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Normalize to [0, 1] range
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Validation pipeline (minimal augmentation)
        self.val_pipeline = A.Compose([
            A.Resize(height=image_size[1], width=image_size[0], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def get_train_pipeline(self) -> A.Compose:
        """Get training augmentation pipeline."""
        return self.train_pipeline
    
    def get_val_pipeline(self) -> A.Compose:
        """Get validation augmentation pipeline."""
        return self.val_pipeline


def create_card_augmentation_config() -> Dict[str, Any]:
    """
    Create configuration dictionary for card-specific augmentations.
    This can be used to configure YOLO's built-in augmentation system.
    """
    return {
        # Geometric augmentations - enhanced for orientation robustness
        'degrees': 60.0,        # Increased rotation range for better orientation handling
        'translate': 0.15,      # Enhanced translation augmentation
        'scale': 0.6,          # Wider scale range
        'shear': 20.0,         # Added shear augmentation for perspective variations
        'perspective': 0.0005,  # Perspective transformation
        
        # Flip augmentations - enhanced for orientation diversity
        'flipud': 0.3,         # Increased vertical flip probability
        'fliplr': 0.5,         # Horizontal flip probability
        
        # Color augmentations - enhanced for lighting robustness
        'hsv_h': 0.02,         # Increased hue augmentation
        'hsv_s': 0.7,          # Saturation augmentation
        'hsv_v': 0.4,          # Value/brightness augmentation
        
        # Advanced composition augmentations
        'mosaic': 1.0,         # Mosaic augmentation
        'mixup': 0.1,          # Mixup augmentation for robustness
        'copy_paste': 0.15,    # Increased copy-paste for data diversity
        
        # Quality and robustness augmentations
        'noise': 0.02,         # Gaussian noise probability
        'blur': 0.01,          # Motion blur probability
        
        # Advanced augmentation techniques
        'auto_augment': 'randaugment',  # Auto augmentation policy
        'erasing': 0.4,        # Random erasing (cutout-style)
        
        # Crop and resize augmentations
        'crop_fraction': 1.0,  # Crop fraction for training
        
        # Environmental and lighting effects
        'brightness_range': 0.2,    # Brightness variation
        'contrast_range': 0.2,      # Contrast variation
        'saturation_range': 0.2,    # Saturation variation
        
        # Advanced blur effects
        'gaussian_blur': 0.05,      # Gaussian blur probability
        'motion_blur': 0.05,        # Motion blur probability
        'median_blur': 0.02,        # Median blur probability
        
        # Environmental simulation
        'shadow': 0.1,              # Shadow effects
        'fog': 0.05,                # Fog effects
        'sun_flare': 0.02,          # Sun flare effects
        
        # Card-specific effects
        'holographic_effect': 0.1,  # Holographic card simulation
        'sleeve_reflection': 0.15,  # Card sleeve reflections
        'surface_glare': 0.2,       # Surface glare and reflections
        
        # Geometric distortions
        'elastic_alpha': 1.0,       # Elastic transformation
        'elastic_sigma': 50.0,      # Elastic transformation sigma
        'optical_distortion': 0.1,  # Optical distortion
        'grid_distortion': 0.1,     # Grid distortion
    }


# Example usage and testing functions
def test_augmentations():
    """Test the augmentation pipelines with sample data."""
    import matplotlib.pyplot as plt
    
    # Create sample image and bounding box
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 150]]  # [x_min, y_min, x_max, y_max]
    class_labels = ['card']
    
    # Test orientation augmentations
    aug = OrientationAugmentations()
    
    for intensity in ['heavy', 'medium', 'light']:
        pipeline = aug.get_pipeline(intensity)
        
        # Apply augmentation
        augmented = pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        print(f"{intensity.capitalize()} augmentation applied successfully")
        print(f"Original bbox: {bboxes[0]}")
        print(f"Augmented bbox: {augmented['bboxes'][0] if augmented['bboxes'] else 'None'}")
        print()


if __name__ == "__main__":
    # Test the augmentation pipelines
    test_augmentations()
    
    # Print configuration for YOLO
    config = create_card_augmentation_config()
    print("YOLO Augmentation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
