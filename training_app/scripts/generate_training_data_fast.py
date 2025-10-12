#!/usr/bin/env python3
"""
Fast parallel training data generation using multiprocessing.
Optimized for maximum CPU utilization and speed.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import albumentations as A
from datetime import datetime
import argparse
from dataclasses import dataclass
from multiprocessing import Pool, Manager, Value, Lock
import time
import psutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SceneConfig:
    """Configuration for scene generation with enhanced orientation support."""
    output_size: Tuple[int, int] = (1024, 768)
    min_cards: int = 1
    max_cards: int = 8
    min_card_size: int = 150
    max_card_size: int = 300
    overlap_probability: float = 0.3
    max_overlap: float = 0.4
    rotation_range: Tuple[float, float] = (-180, 180)  # Full rotation range for orientation robustness
    perspective_probability: float = 0.4
    lighting_probability: float = 0.6
    
    # Enhanced orientation settings
    common_orientation_bias: float = 0.6  # Probability of using common orientations (0°, 90°, 180°, 270°)
    orientation_variation: float = 15.0   # Variation around common orientations in degrees


# Global variables for multiprocessing
CARD_IMAGES_LIST = None
CONFIG = None
OUTPUT_DIR = None


def init_worker(card_images_list, config, output_dir):
    """Initialize worker process with shared data."""
    global CARD_IMAGES_LIST, CONFIG, OUTPUT_DIR
    CARD_IMAGES_LIST = card_images_list
    CONFIG = config
    OUTPUT_DIR = output_dir


def generate_background(size: Tuple[int, int]) -> np.ndarray:
    """Generate a synthetic background."""
    width, height = size
    background_type = random.choice(['solid', 'gradient', 'texture', 'wood', 'fabric'])
    
    if background_type == 'solid':
        color = [random.randint(200, 255) for _ in range(3)]
        background = np.full((height, width, 3), color, dtype=np.uint8)
        
    elif background_type == 'gradient':
        background = np.zeros((height, width, 3), dtype=np.uint8)
        color1 = [random.randint(180, 255) for _ in range(3)]
        color2 = [random.randint(180, 255) for _ in range(3)]
        
        for i in range(height):
            ratio = i / height
            for j in range(3):
                background[i, :, j] = int(color1[j] * (1 - ratio) + color2[j] * ratio)
                
    elif background_type == 'texture':
        background = np.random.randint(200, 240, (height, width, 3), dtype=np.uint8)
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif background_type == 'wood':
        base_color = [139, 115, 85]
        background = np.full((height, width, 3), base_color, dtype=np.uint8)
        for i in range(0, width, 20):
            grain_intensity = random.randint(-30, 30)
            background[:, i:i+10] = np.clip(
                background[:, i:i+10].astype(np.int16) + grain_intensity, 0, 255
            ).astype(np.uint8)
            
    else:  # fabric
        base_colors = [[220, 220, 220], [200, 200, 220], [220, 200, 200]]
        base_color = random.choice(base_colors)
        background = np.full((height, width, 3), base_color, dtype=np.uint8)
        
        for i in range(0, height, 5):
            for j in range(0, width, 5):
                variation = random.randint(-15, 15)
                background[i:i+5, j:j+5] = np.clip(
                    background[i:i+5, j:j+5].astype(np.int16) + variation, 0, 255
                ).astype(np.uint8)
    
    return background


def transform_card(card_image: np.ndarray, rotation: float, scale: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Transform a card image with rotation and scaling using enhanced albumentations."""
    
    # Enhanced albumentations pipeline for individual cards
    card_augmentation = A.Compose([
        # Photometric augmentations with higher variation for robustness
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        
        # Color variations for different lighting conditions
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.4),
        
        # Noise for robustness to image quality variations
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
        ], p=0.3),
        
        # Subtle blur effects that might occur in real photos
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.15),
        
    ])
    
    # Apply albumentations
    try:
        augmented = card_augmentation(image=card_image)
        card_image = augmented['image']
    except Exception as e:
        # Fallback to simple augmentations if albumentations fails
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.2)
            card_image = np.clip(card_image * brightness, 0, 255).astype(np.uint8)
        
        if random.random() < 0.5:
            contrast = random.uniform(0.9, 1.1)
            card_image = np.clip((card_image - 128) * contrast + 128, 0, 255).astype(np.uint8)
    
    height, width = card_image.shape[:2]
    
    # Scale the card
    new_width = int(width * scale)
    new_height = int(height * scale)
    card_image = cv2.resize(card_image, (new_width, new_height))
    
    # Store original scaled dimensions (before rotation)
    original_scaled_size = (new_width, new_height)
    
    # Create rotation matrix
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    
    # Calculate new bounding box after rotation
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int((new_height * sin_val) + (new_width * cos_val))
    new_h = int((new_height * cos_val) + (new_width * sin_val))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    rotated_card = cv2.warpAffine(card_image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    
    # Create mask
    mask = cv2.warpAffine(np.ones((new_height, new_width), dtype=np.uint8) * 255, 
                         rotation_matrix, (new_w, new_h),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    
    # Store transformation info for proper bbox calculation
    transform_info = {
        'rotation': rotation,
        'scale': scale,
        'original_scaled_size': original_scaled_size,
        'rotation_matrix': rotation_matrix,
        'rotated_size': (new_w, new_h)
    }
    
    return rotated_card, mask, transform_info


def calculate_rotated_bbox(center_x: float, center_y: float, width: float, height: float, 
                          rotation_degrees: float) -> List[List[float]]:
    """Calculate rotated bounding box corners."""
    # Convert rotation to radians
    rotation_rad = np.radians(rotation_degrees)
    
    # Half dimensions
    half_w = width / 2
    half_h = height / 2
    
    # Original corners relative to center
    corners = np.array([
        [-half_w, -half_h],  # top-left
        [half_w, -half_h],   # top-right
        [half_w, half_h],    # bottom-right
        [-half_w, half_h]    # bottom-left
    ])
    
    # Rotation matrix
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # Rotate corners
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to actual position
    rotated_corners[:, 0] += center_x
    rotated_corners[:, 1] += center_y
    
    return rotated_corners.tolist()


def find_card_position(card_shape: Tuple[int, int], occupied_regions: List[Tuple], 
                      scene_size: Tuple[int, int], allow_overlap: bool = False) -> Tuple[int, int]:
    """Find a valid position for placing a card."""
    card_h, card_w = card_shape
    scene_w, scene_h = scene_size
    
    max_attempts = 30
    for _ in range(max_attempts):
        x = random.randint(0, max(0, scene_w - card_w))
        y = random.randint(0, max(0, scene_h - card_h))
        
        if not occupied_regions:
            return x, y
        
        card_rect = (x, y, x + card_w, y + card_h)
        
        # Check for overlaps
        overlap_ratio = 0
        for occupied in occupied_regions:
            x1_1, y1_1, x2_1, y2_1 = card_rect
            x1_2, y1_2, x2_2, y2_2 = occupied
            
            x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            
            if x_overlap > 0 and y_overlap > 0:
                intersection_area = x_overlap * y_overlap
                rect1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                overlap_ratio = max(overlap_ratio, intersection_area / rect1_area if rect1_area > 0 else 0)
        
        if not allow_overlap and overlap_ratio == 0:
            return x, y
        elif allow_overlap and overlap_ratio <= CONFIG.max_overlap:
            return x, y
    
    return None


def generate_single_background(args):
    """Generate a single background image without cards - optimized for multiprocessing."""
    bg_index, split = args
    
    try:
        # Generate background
        background = generate_background(CONFIG.output_size)
        
        # Apply scene-level effects using albumentations (same as full scenes)
        scene_augmentation = A.Compose([
            # Lighting and exposure variations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=CONFIG.lighting_probability),
            
            # Color temperature variations
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ], p=0.3),
            
            # Environmental effects
            A.OneOf([
                A.GaussNoise(var_limit=(3.0, 15.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.2), p=1.0),
            ], p=0.2),
            
            # Slight blur
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=2, p=1.0),
            ], p=0.1),
        ])
        
        try:
            augmented_scene = scene_augmentation(image=background)
            background = augmented_scene['image']
        except Exception as e:
            # Fallback to simple brightness adjustment
            if random.random() < CONFIG.lighting_probability:
                brightness_factor = random.uniform(0.8, 1.2)
                background = np.clip(background * brightness_factor, 0, 255).astype(np.uint8)
        
        # Save image with distinct naming pattern
        image_filename = f"{split}_bg_{bg_index:06d}.jpg"
        image_path = Path(OUTPUT_DIR) / "images" / image_filename
        cv2.imwrite(str(image_path), background)
        
        return {
            'bg_index': bg_index,
            'image_filename': image_filename,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error generating background {bg_index}: {e}")
        return {'bg_index': bg_index, 'success': False, 'error': str(e)}


def generate_single_scene(args):
    """Generate a single scene - optimized for multiprocessing."""
    scene_index, split = args
    
    try:
        # Generate background
        background = generate_background(CONFIG.output_size)
        scene = background.copy()
        
        # Determine number of cards
        num_cards = random.randint(CONFIG.min_cards, CONFIG.max_cards)
        
        # Select random cards
        selected_cards = random.sample(CARD_IMAGES_LIST, min(num_cards, len(CARD_IMAGES_LIST)))
        
        card_instances = []
        occupied_regions = []
        
        for i, card_path in enumerate(selected_cards):
            # Load card image
            card_image = cv2.imread(str(card_path))
            if card_image is None:
                continue
            
            # Generate enhanced transformation parameters for orientation robustness
            scale = random.uniform(
                CONFIG.min_card_size / max(card_image.shape[:2]),
                CONFIG.max_card_size / max(card_image.shape[:2])
            )
            
            # Enhanced rotation range with bias towards common orientations
            if random.random() < CONFIG.common_orientation_bias:
                # Common orientations (0°, 90°, 180°, 270°) with small variations
                base_angles = [0, 90, 180, 270]
                base_angle = random.choice(base_angles)
                rotation = base_angle + random.uniform(-CONFIG.orientation_variation, CONFIG.orientation_variation)
            else:
                # Random orientations for full coverage
                rotation = random.uniform(*CONFIG.rotation_range)
            
            # Transform card
            transformed_card, card_mask, transform_info = transform_card(card_image, rotation, scale)
            
            # Find position for the card
            position = find_card_position(
                transformed_card.shape[:2], occupied_regions, CONFIG.output_size,
                allow_overlap=random.random() < CONFIG.overlap_probability
            )
            
            if position is None:
                continue
            
            x, y = position
            card_h, card_w = transformed_card.shape[:2]
            
            # Ensure card fits in scene
            if x + card_w > CONFIG.output_size[0] or y + card_h > CONFIG.output_size[1]:
                continue
            
            # Place card in scene using mask
            card_region = scene[y:y+card_h, x:x+card_w]
            mask_3d = np.stack([card_mask] * 3, axis=2) / 255.0
            
            # Blend card with background
            scene[y:y+card_h, x:x+card_w] = (
                transformed_card * mask_3d + card_region * (1 - mask_3d)
            ).astype(np.uint8)
            
            # Store card instance info
            card_instances.append({
                'bbox': (x, y, card_w, card_h),
                'mask': card_mask,
                'card_path': str(card_path),
                'transform_info': transform_info,
                'position': (x, y)
            })
            
            # Update occupied regions
            occupied_regions.append((x, y, x + card_w, y + card_h))
        
        # Apply enhanced scene-level effects using albumentations
        scene_augmentation = A.Compose([
            # Lighting and exposure variations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=CONFIG.lighting_probability),
            
            # Color temperature variations (simulating different lighting conditions)
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ], p=0.3),
            
            # Environmental effects
            A.OneOf([
                A.GaussNoise(var_limit=(3.0, 15.0), p=1.0),  # Camera sensor noise
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.2), p=1.0),  # ISO noise
            ], p=0.2),
            
            # Slight blur from camera shake or motion
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=2, p=1.0),
            ], p=0.1),
        ])
        
        try:
            augmented_scene = scene_augmentation(image=scene)
            scene = augmented_scene['image']
        except Exception as e:
            # Fallback to simple brightness adjustment
            if random.random() < CONFIG.lighting_probability:
                brightness_factor = random.uniform(0.8, 1.2)
                scene = np.clip(scene * brightness_factor, 0, 255).astype(np.uint8)
        
        # Save image
        image_filename = f"{split}_{scene_index:06d}.jpg"
        image_path = Path(OUTPUT_DIR) / "images" / image_filename
        cv2.imwrite(str(image_path), scene)
        
        # Generate annotations with occlusion handling
        annotations = []
        
        for i, card_instance in enumerate(card_instances):
            x, y, w, h = card_instance['bbox']
            mask = card_instance['mask']
            transform_info = card_instance['transform_info']
            
            # Create segmentation mask for this card
            scene_mask = np.zeros(CONFIG.output_size[::-1], dtype=np.uint8)  # height, width
            card_h, card_w = mask.shape
            
            end_y = min(y + card_h, CONFIG.output_size[1])
            end_x = min(x + card_w, CONFIG.output_size[0])
            actual_h = end_y - y
            actual_w = end_x - x
            
            if actual_h > 0 and actual_w > 0:
                # Place this card's mask
                card_mask_region = mask[:actual_h, :actual_w]
                scene_mask[y:end_y, x:end_x] = card_mask_region
                
                # Remove occluded parts by checking against later cards (higher Z-order)
                for j in range(i + 1, len(card_instances)):
                    later_card = card_instances[j]
                    later_x, later_y, later_w, later_h = later_card['bbox']
                    later_mask = later_card['mask']
                    
                    # Check if there's overlap
                    overlap_x1 = max(x, later_x)
                    overlap_y1 = max(y, later_y)
                    overlap_x2 = min(x + w, later_x + later_w)
                    overlap_y2 = min(y + h, later_y + later_h)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        # There's overlap - remove occluded parts
                        # Calculate overlap region in scene coordinates
                        overlap_w = overlap_x2 - overlap_x1
                        overlap_h = overlap_y2 - overlap_y1
                        
                        # Get the later card's mask in the overlap region
                        later_mask_x1 = overlap_x1 - later_x
                        later_mask_y1 = overlap_y1 - later_y
                        later_mask_x2 = later_mask_x1 + overlap_w
                        later_mask_y2 = later_mask_y1 + overlap_h
                        
                        # Ensure we don't go out of bounds
                        later_mask_h, later_mask_w = later_mask.shape
                        later_mask_x2 = min(later_mask_x2, later_mask_w)
                        later_mask_y2 = min(later_mask_y2, later_mask_h)
                        
                        if (later_mask_x2 > later_mask_x1 and later_mask_y2 > later_mask_y1):
                            later_overlap_mask = later_mask[later_mask_y1:later_mask_y2, 
                                                           later_mask_x1:later_mask_x2]
                            
                            # Remove occluded pixels from current card's mask
                            occlusion_pixels = later_overlap_mask > 128
                            scene_mask[overlap_y1:overlap_y2, overlap_x1:overlap_x2][occlusion_pixels] = 0
            
            # Find contours for segmentation
            contours, _ = cv2.findContours(scene_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Skip cards that are completely occluded (no visible area)
            if contours and np.sum(scene_mask > 128) > 100:  # Minimum 100 visible pixels
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(simplified_contour) >= 3:
                    segmentation = simplified_contour.flatten().tolist()
                    area = cv2.contourArea(largest_contour)
                    
                    # Calculate rotated bounding box
                    original_w, original_h = transform_info['original_scaled_size']
                    rotation = transform_info['rotation']
                    
                    # Calculate center of the ACTUAL card (not the rotated bounding box)
                    # The rotated card is centered within the larger rotated bounding box
                    # We need to find where the original card center is in scene coordinates
                    rotation_matrix = transform_info['rotation_matrix']
                    
                    # The center of the original card in the rotated image
                    rotated_size = transform_info['rotated_size']
                    card_center_in_rotated = (rotated_size[0] / 2, rotated_size[1] / 2)
                    
                    # Transform this to scene coordinates
                    center_x = x + card_center_in_rotated[0]
                    center_y = y + card_center_in_rotated[1]
                    
                    # Get rotated bounding box corners
                    # Note: OpenCV uses clockwise positive, math uses counter-clockwise positive
                    # So we negate the rotation to match OpenCV's direction
                    rotated_corners = calculate_rotated_bbox(center_x, center_y, original_w, original_h, -rotation)
                    
                    # Flatten corners for segmentation format
                    rotated_bbox_segmentation = []
                    for corner in rotated_corners:
                        rotated_bbox_segmentation.extend([float(corner[0]), float(corner[1])])
                    
                    # Calculate axis-aligned bbox for compatibility (min enclosing rectangle)
                    x_coords = [corner[0] for corner in rotated_corners]
                    y_coords = [corner[1] for corner in rotated_corners]
                    axis_aligned_bbox = [
                        int(min(x_coords)), int(min(y_coords)), 
                        int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                    ]
                    
                    annotations.append({
                        'segmentation': [segmentation],
                        'area': area,
                        'bbox': axis_aligned_bbox,
                        'rotated_bbox': rotated_bbox_segmentation,  # Add rotated bbox
                        'rotation': rotation,  # Store rotation for reference
                        'original_size': [original_w, original_h]  # Store original card size
                    })
        
        return {
            'scene_index': scene_index,
            'image_filename': image_filename,
            'annotations': annotations,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error generating scene {scene_index}: {e}")
        return {'scene_index': scene_index, 'success': False, 'error': str(e)}


def generate_backgrounds_fast(output_dir: str, num_train: int, num_val: int, num_test: int,
                             config: SceneConfig, num_workers: int = None):
    """Generate background images only using fast multiprocessing."""
    
    # Determine optimal worker count
    if num_workers is None:
        cpu_count = psutil.cpu_count(logical=True)
        num_workers = min(cpu_count, 16)  # Cap at 16 workers
    
    logger.info(f"Using {num_workers} workers for background generation")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate each split
    splits = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]
    
    total_generated = 0
    
    for split_name, num_images in splits:
        if num_images == 0:
            continue
            
        logger.info(f"Generating {num_images:,} {split_name} background images...")
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        args_list = [(i, split_name) for i in range(num_images)]
        
        # Process in parallel
        with Pool(processes=num_workers, initializer=init_worker, 
                 initargs=([], config, output_dir)) as pool:
            
            results = []
            completed = 0
            
            # Process in chunks for progress reporting
            chunk_size = min(1000, max(100, num_images // 20))
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i+chunk_size]
                chunk_results = pool.map(generate_single_background, chunk)
                results.extend(chunk_results)
                
                completed += len(chunk)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = num_images - completed
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {completed:,}/{num_images:,} ({completed/num_images*100:.1f}%) "
                          f"- Rate: {rate:.1f} img/sec - ETA: {eta/60:.1f} min")
        
        # Process results
        successful_results = [r for r in results if r.get('success', False)]
        
        elapsed_time = time.time() - start_time
        images_per_second = len(successful_results) / elapsed_time
        
        logger.info(f"{split_name.capitalize()} complete: {len(successful_results):,} background images "
                   f"in {elapsed_time:.1f}s ({images_per_second:.1f} img/sec)")
        
        total_generated += len(successful_results)
    
    return total_generated


def generate_dataset_fast(card_images_dir: str, output_dir: str, num_train: int, num_val: int, num_test: int, 
                         config: SceneConfig, num_workers: int = None):
    """Generate dataset using fast multiprocessing."""
    
    # Determine optimal worker count
    if num_workers is None:
        cpu_count = psutil.cpu_count(logical=True)
        num_workers = min(cpu_count, 16)  # Cap at 16 workers
    
    logger.info(f"Using {num_workers} workers for generation")
    
    # Pre-load card images list
    card_images_dir = Path(card_images_dir)
    card_images = []
    for game_dir in ['mtg', 'yugioh', 'pokemon']:
        game_path = card_images_dir / game_dir
        if game_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                card_images.extend(list(game_path.glob(ext)))
    
    logger.info(f"Pre-loaded {len(card_images)} card images")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate each split
    splits = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]
    
    for split_name, num_images in splits:
        if num_images == 0:
            continue
            
        logger.info(f"Generating {num_images:,} {split_name} images...")
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        args_list = [(i, split_name) for i in range(num_images)]
        
        # Process in parallel
        with Pool(processes=num_workers, initializer=init_worker, 
                 initargs=(card_images, config, output_dir)) as pool:
            
            results = []
            completed = 0
            
            # Process in chunks for progress reporting
            chunk_size = min(1000, max(100, num_images // 20))
            
            for i in range(0, len(args_list), chunk_size):
                chunk = args_list[i:i+chunk_size]
                chunk_results = pool.map(generate_single_scene, chunk)
                results.extend(chunk_results)
                
                completed += len(chunk)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = num_images - completed
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {completed:,}/{num_images:,} ({completed/num_images*100:.1f}%) "
                          f"- Rate: {rate:.1f} img/sec - ETA: {eta/60:.1f} min")
        
        # Process results and create COCO annotations
        successful_results = [r for r in results if r.get('success', False)]
        
        coco_dataset = {
            "info": {
                "description": "Trading Card Segmentation Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "Trading Card Segmentation Project",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "card", "supercategory": "object"}]
        }
        
        annotation_id = 1
        for image_id, result in enumerate(successful_results, 1):
            # Add image info
            coco_dataset["images"].append({
                "id": image_id,
                "width": config.output_size[0],
                "height": config.output_size[1],
                "file_name": result['image_filename']
            })
            
            # Add annotations
            for ann in result['annotations']:
                annotation_data = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": ann['segmentation'],
                    "area": ann['area'],
                    "bbox": ann['bbox'],
                    "iscrowd": 0
                }
                
                # Add rotated bounding box information if available
                if 'rotated_bbox' in ann:
                    annotation_data["rotated_bbox"] = ann['rotated_bbox']
                    annotation_data["rotation"] = ann['rotation']
                    annotation_data["original_size"] = ann['original_size']
                
                coco_dataset["annotations"].append(annotation_data)
                annotation_id += 1
        
        # Save annotations
        annotation_file = annotations_dir / f"{split_name}_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
        
        elapsed_time = time.time() - start_time
        images_per_second = len(successful_results) / elapsed_time
        
        logger.info(f"{split_name.capitalize()} complete: {len(successful_results):,} images, "
                   f"{len(coco_dataset['annotations']):,} annotations in {elapsed_time:.1f}s "
                   f"({images_per_second:.1f} img/sec)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast parallel training data generation")
    parser.add_argument("--card_images_dir", type=str, default="data/raw/card_images")
    parser.add_argument("--output_dir", type=str, default="data/training_fast")
    parser.add_argument("--num_train", type=int, default=35000)
    parser.add_argument("--num_val", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=5000)
    parser.add_argument("--min_cards", type=int, default=1)
    parser.add_argument("--max_cards", type=int, default=8)
    parser.add_argument("--output_size", type=str, default="1024,768")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0=auto)")
    
    # Background generation arguments
    parser.add_argument("--generate_backgrounds", action="store_true", 
                       help="Generate only background images without cards")
    parser.add_argument("--num_bg_train", type=int, default=1000,
                       help="Number of training background images (only used with --generate_backgrounds)")
    parser.add_argument("--num_bg_val", type=int, default=300,
                       help="Number of validation background images (only used with --generate_backgrounds)")
    parser.add_argument("--num_bg_test", type=int, default=200,
                       help="Number of test background images (only used with --generate_backgrounds)")
    
    args = parser.parse_args()
    
    # Parse output size
    width, height = map(int, args.output_size.split(','))
    
    # Create config
    config = SceneConfig(
        output_size=(width, height),
        min_cards=args.min_cards,
        max_cards=args.max_cards
    )
    
    # Determine worker count
    num_workers = args.workers if args.workers > 0 else None
    
    start_time = time.time()
    
    if args.generate_backgrounds:
        # Generate backgrounds only
        logger.info("Background generation mode enabled")
        total_images = generate_backgrounds_fast(
            args.output_dir,
            args.num_bg_train, args.num_bg_val, args.num_bg_test,
            config, num_workers
        )
        
        total_time = time.time() - start_time
        
        print(f"\nBackground Generation Complete!")
        print(f"Generated {total_images:,} background images in {total_time/60:.1f} minutes")
        print(f"Average rate: {total_images/total_time:.1f} images/sec")
        print(f"Output: {args.output_dir}/images/")
        print(f"Naming pattern: {{split}}_bg_{{index:06d}}.jpg")
    else:
        # Generate full dataset with cards
        generate_dataset_fast(
            args.card_images_dir, args.output_dir,
            args.num_train, args.num_val, args.num_test,
            config, num_workers
        )
        
        total_time = time.time() - start_time
        total_images = args.num_train + args.num_val + args.num_test
        
        print(f"\nFast Generation Complete!")
        print(f"Generated {total_images:,} images in {total_time/60:.1f} minutes")
        print(f"Average rate: {total_images/total_time:.1f} images/sec")
        print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
