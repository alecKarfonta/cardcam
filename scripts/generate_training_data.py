#!/usr/bin/env python3
"""
Generate segmentation training examples from individual card images.
Creates synthetic multi-card scenes with proper annotations for training.
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
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CardInstance:
    """Represents a card instance in a scene."""
    image_path: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    mask: np.ndarray
    rotation: float
    scale: float
    card_id: str


@dataclass
class SceneConfig:
    """Configuration for scene generation."""
    output_size: Tuple[int, int] = (1024, 768)
    min_cards: int = 1
    max_cards: int = 8
    min_card_size: int = 150
    max_card_size: int = 300
    overlap_probability: float = 0.3
    max_overlap: float = 0.4
    rotation_range: Tuple[float, float] = (-15, 15)
    perspective_probability: float = 0.4
    lighting_probability: float = 0.6


class BackgroundGenerator:
    """Generates realistic backgrounds for card scenes."""
    
    def __init__(self, background_dir: str = None):
        self.background_dir = Path(background_dir) if background_dir else None
        self.background_images = []
        
        if self.background_dir and self.background_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.background_images.extend(list(self.background_dir.glob(ext)))
    
    def generate_background(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate a background image."""
        width, height = size
        
        if self.background_images and random.random() < 0.7:
            # Use real background image
            bg_path = random.choice(self.background_images)
            background = cv2.imread(str(bg_path))
            background = cv2.resize(background, (width, height))
        else:
            # Generate synthetic background
            background = self._generate_synthetic_background(width, height)
        
        return background
    
    def _generate_synthetic_background(self, width: int, height: int) -> np.ndarray:
        """Generate synthetic background patterns."""
        background_type = random.choice(['solid', 'gradient', 'texture', 'wood', 'fabric'])
        
        if background_type == 'solid':
            # Solid color background
            color = [random.randint(200, 255) for _ in range(3)]  # Light colors
            background = np.full((height, width, 3), color, dtype=np.uint8)
            
        elif background_type == 'gradient':
            # Gradient background
            background = np.zeros((height, width, 3), dtype=np.uint8)
            color1 = [random.randint(180, 255) for _ in range(3)]
            color2 = [random.randint(180, 255) for _ in range(3)]
            
            for i in range(height):
                ratio = i / height
                for j in range(3):
                    background[i, :, j] = int(color1[j] * (1 - ratio) + color2[j] * ratio)
                    
        elif background_type == 'texture':
            # Textured background (noise-based)
            background = np.random.randint(200, 240, (height, width, 3), dtype=np.uint8)
            noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
            background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif background_type == 'wood':
            # Wood-like pattern
            base_color = [139, 115, 85]  # Brown
            background = np.full((height, width, 3), base_color, dtype=np.uint8)
            
            # Add wood grain
            for i in range(0, width, 20):
                grain_intensity = random.randint(-30, 30)
                background[:, i:i+10] = np.clip(
                    background[:, i:i+10].astype(np.int16) + grain_intensity, 0, 255
                ).astype(np.uint8)
                
        else:  # fabric
            # Fabric-like pattern
            base_colors = [[220, 220, 220], [200, 200, 220], [220, 200, 200]]
            base_color = random.choice(base_colors)
            background = np.full((height, width, 3), base_color, dtype=np.uint8)
            
            # Add fabric texture
            for i in range(0, height, 5):
                for j in range(0, width, 5):
                    variation = random.randint(-15, 15)
                    background[i:i+5, j:j+5] = np.clip(
                        background[i:i+5, j:j+5].astype(np.int16) + variation, 0, 255
                    ).astype(np.uint8)
        
        return background


class CardTransformer:
    """Handles card transformations and augmentations."""
    
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ])
    
    def transform_card(self, card_image: np.ndarray, rotation: float, scale: float, 
                      perspective: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Transform a card image with rotation, scaling, and perspective."""
        height, width = card_image.shape[:2]
        
        # Apply augmentations
        augmented = self.augmentation_pipeline(image=card_image)
        card_image = augmented['image']
        
        # Scale the card
        new_width = int(width * scale)
        new_height = int(height * scale)
        card_image = cv2.resize(card_image, (new_width, new_height))
        
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
        
        # Create mask (non-zero pixels)
        mask = cv2.warpAffine(np.ones((new_height, new_width), dtype=np.uint8) * 255, 
                             rotation_matrix, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
        
        # Apply perspective transformation if requested
        if perspective and random.random() < 0.5:
            rotated_card, mask = self._apply_perspective(rotated_card, mask)
        
        return rotated_card, mask
    
    def _apply_perspective(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective transformation to simulate 3D rotation."""
        height, width = image.shape[:2]
        
        # Define perspective transformation points
        perspective_strength = random.uniform(0.1, 0.3)
        
        # Original corners
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Distorted corners
        dst_points = np.float32([
            [random.uniform(0, width * perspective_strength), 
             random.uniform(0, height * perspective_strength)],
            [width - random.uniform(0, width * perspective_strength), 
             random.uniform(0, height * perspective_strength)],
            [width - random.uniform(0, width * perspective_strength), 
             height - random.uniform(0, height * perspective_strength)],
            [random.uniform(0, width * perspective_strength), 
             height - random.uniform(0, height * perspective_strength)]
        ])
        
        # Apply perspective transformation
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        transformed_image = cv2.warpPerspective(image, perspective_matrix, (width, height),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
        
        transformed_mask = cv2.warpPerspective(mask, perspective_matrix, (width, height),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)
        
        return transformed_image, transformed_mask


class SceneGenerator:
    """Generates multi-card scenes for training."""
    
    def __init__(self, card_images_dir: str, config: SceneConfig, background_generator: BackgroundGenerator):
        self.card_images_dir = Path(card_images_dir)
        self.config = config
        self.background_generator = background_generator
        self.transformer = CardTransformer()
        
        # Load available card images
        self.card_images = []
        for game_dir in ['mtg', 'yugioh', 'pokemon']:
            game_path = self.card_images_dir / game_dir
            if game_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    self.card_images.extend(list(game_path.glob(ext)))
        
        logger.info(f"Found {len(self.card_images)} card images")
    
    def generate_scene(self) -> Tuple[np.ndarray, List[CardInstance]]:
        """Generate a single multi-card scene."""
        # Generate background
        background = self.background_generator.generate_background(self.config.output_size)
        scene = background.copy()
        
        # Determine number of cards
        num_cards = random.randint(self.config.min_cards, self.config.max_cards)
        
        # Select random cards
        selected_cards = random.sample(self.card_images, min(num_cards, len(self.card_images)))
        
        card_instances = []
        occupied_regions = []  # Track occupied areas for overlap detection
        
        for i, card_path in enumerate(selected_cards):
            # Load card image
            card_image = cv2.imread(str(card_path))
            if card_image is None:
                continue
            
            # Generate transformation parameters
            scale = random.uniform(
                self.config.min_card_size / max(card_image.shape[:2]),
                self.config.max_card_size / max(card_image.shape[:2])
            )
            rotation = random.uniform(*self.config.rotation_range)
            
            # Transform card
            transformed_card, card_mask = self.transformer.transform_card(
                card_image, rotation, scale, 
                perspective=random.random() < self.config.perspective_probability
            )
            
            # Find position for the card
            position = self._find_card_position(
                transformed_card.shape[:2], occupied_regions, 
                allow_overlap=random.random() < self.config.overlap_probability
            )
            
            if position is None:
                continue  # Skip if no valid position found
            
            x, y = position
            card_h, card_w = transformed_card.shape[:2]
            
            # Ensure card fits in scene
            if x + card_w > self.config.output_size[0] or y + card_h > self.config.output_size[1]:
                continue
            
            # Place card in scene using mask
            card_region = scene[y:y+card_h, x:x+card_w]
            mask_3d = np.stack([card_mask] * 3, axis=2) / 255.0
            
            # Blend card with background
            scene[y:y+card_h, x:x+card_w] = (
                transformed_card * mask_3d + card_region * (1 - mask_3d)
            ).astype(np.uint8)
            
            # Create card instance
            card_instance = CardInstance(
                image_path=str(card_path),
                bbox=(x, y, card_w, card_h),
                mask=card_mask,
                rotation=rotation,
                scale=scale,
                card_id=f"card_{i}_{card_path.stem}"
            )
            card_instances.append(card_instance)
            
            # Update occupied regions
            occupied_regions.append((x, y, x + card_w, y + card_h))
        
        # Apply scene-level augmentations
        scene = self._apply_scene_augmentations(scene)
        
        return scene, card_instances
    
    def _find_card_position(self, card_shape: Tuple[int, int], occupied_regions: List[Tuple], 
                           allow_overlap: bool = False) -> Tuple[int, int]:
        """Find a valid position for placing a card."""
        card_h, card_w = card_shape
        scene_w, scene_h = self.config.output_size
        
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(0, max(0, scene_w - card_w))
            y = random.randint(0, max(0, scene_h - card_h))
            
            card_rect = (x, y, x + card_w, y + card_h)
            
            if not occupied_regions:
                return x, y
            
            # Check for overlaps
            overlap_ratio = 0
            for occupied in occupied_regions:
                overlap = self._calculate_overlap(card_rect, occupied)
                overlap_ratio = max(overlap_ratio, overlap)
            
            if not allow_overlap and overlap_ratio == 0:
                return x, y
            elif allow_overlap and overlap_ratio <= self.config.max_overlap:
                return x, y
        
        return None  # No valid position found
    
    def _calculate_overlap(self, rect1: Tuple, rect2: Tuple) -> float:
        """Calculate overlap ratio between two rectangles."""
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2
        
        # Calculate intersection
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        
        if x_overlap == 0 or y_overlap == 0:
            return 0
        
        intersection_area = x_overlap * y_overlap
        rect1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection_area / rect1_area if rect1_area > 0 else 0
    
    def _apply_scene_augmentations(self, scene: np.ndarray) -> np.ndarray:
        """Apply scene-level augmentations."""
        if random.random() < self.config.lighting_probability:
            # Adjust lighting
            scene = self._adjust_lighting(scene)
        
        # Add slight blur occasionally
        if random.random() < 0.1:
            scene = cv2.GaussianBlur(scene, (3, 3), 0)
        
        return scene
    
    def _adjust_lighting(self, scene: np.ndarray) -> np.ndarray:
        """Apply realistic lighting effects."""
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # Random contrast adjustment
        contrast_factor = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class AnnotationGenerator:
    """Generates COCO format annotations."""
    
    def __init__(self):
        self.annotation_id = 1
        self.image_id = 1
    
    def generate_coco_annotation(self, scene: np.ndarray, card_instances: List[CardInstance], 
                               image_filename: str) -> Dict[str, Any]:
        """Generate COCO format annotation for a scene."""
        height, width = scene.shape[:2]
        
        # Image info
        image_info = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": image_filename
        }
        
        # Annotations for each card
        annotations = []
        for card_instance in card_instances:
            x, y, w, h = card_instance.bbox
            
            # Create segmentation mask
            mask = np.zeros((height, width), dtype=np.uint8)
            card_h, card_w = card_instance.mask.shape
            
            # Place card mask in scene coordinates
            end_y = min(y + card_h, height)
            end_x = min(x + card_w, width)
            actual_h = end_y - y
            actual_w = end_x - x
            
            if actual_h > 0 and actual_w > 0:
                mask[y:end_y, x:end_x] = card_instance.mask[:actual_h, :actual_w]
            
            # Convert mask to polygon (simplified)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Convert to segmentation format
                segmentation = simplified_contour.flatten().tolist()
                
                # Calculate area
                area = cv2.contourArea(largest_contour)
                
                # Calculate bounding box from contour
                x_coords = simplified_contour[:, 0, 0]
                y_coords = simplified_contour[:, 0, 1]
                bbox = [int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))]
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": 1,  # Card category
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                annotations.append(annotation)
                self.annotation_id += 1
        
        self.image_id += 1
        
        return {
            "image": image_info,
            "annotations": annotations
        }


class TrainingDataGenerator:
    """Main class for generating training data."""
    
    def __init__(self, card_images_dir: str, output_dir: str, config: SceneConfig = None):
        self.card_images_dir = Path(card_images_dir)
        self.output_dir = Path(output_dir)
        self.config = config or SceneConfig()
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.background_generator = BackgroundGenerator()
        self.scene_generator = SceneGenerator(card_images_dir, self.config, self.background_generator)
        self.annotation_generator = AnnotationGenerator()
        
        # COCO dataset structure
        self.coco_dataset = {
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
            "categories": [
                {
                    "id": 1,
                    "name": "card",
                    "supercategory": "object"
                }
            ]
        }
    
    def generate_dataset(self, num_images: int, split: str = "train"):
        """Generate a complete dataset."""
        logger.info(f"Generating {num_images} {split} images...")
        
        for i in range(num_images):
            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_images} images")
            
            # Generate scene
            scene, card_instances = self.scene_generator.generate_scene()
            
            # Save image
            image_filename = f"{split}_{i:06d}.jpg"
            image_path = self.images_dir / image_filename
            cv2.imwrite(str(image_path), scene)
            
            # Generate annotation
            annotation_data = self.annotation_generator.generate_coco_annotation(
                scene, card_instances, image_filename
            )
            
            # Add to COCO dataset
            self.coco_dataset["images"].append(annotation_data["image"])
            self.coco_dataset["annotations"].extend(annotation_data["annotations"])
        
        # Save COCO annotations
        annotation_file = self.annotations_dir / f"{split}_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)
        
        logger.info(f"Dataset generation complete!")
        logger.info(f"Images saved to: {self.images_dir}")
        logger.info(f"Annotations saved to: {annotation_file}")
        
        return {
            "images_dir": str(self.images_dir),
            "annotations_file": str(annotation_file),
            "num_images": num_images,
            "num_annotations": len(self.coco_dataset["annotations"])
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate segmentation training data")
    parser.add_argument("--card_images_dir", type=str, default="data/raw/card_images",
                       help="Directory containing card images")
    parser.add_argument("--output_dir", type=str, default="data/training_scenes",
                       help="Output directory for generated scenes")
    parser.add_argument("--num_train", type=int, default=1000,
                       help="Number of training images to generate")
    parser.add_argument("--num_val", type=int, default=200,
                       help="Number of validation images to generate")
    parser.add_argument("--min_cards", type=int, default=1,
                       help="Minimum cards per scene")
    parser.add_argument("--max_cards", type=int, default=6,
                       help="Maximum cards per scene")
    parser.add_argument("--output_size", type=str, default="1024,768",
                       help="Output image size (width,height)")
    
    args = parser.parse_args()
    
    # Parse output size
    width, height = map(int, args.output_size.split(','))
    
    # Create config
    config = SceneConfig(
        output_size=(width, height),
        min_cards=args.min_cards,
        max_cards=args.max_cards
    )
    
    # Initialize generator
    generator = TrainingDataGenerator(args.card_images_dir, args.output_dir, config)
    
    # Generate training data
    train_results = generator.generate_dataset(args.num_train, "train")
    
    # Reset for validation data
    generator.annotation_generator.image_id = 1
    generator.annotation_generator.annotation_id = 1
    generator.coco_dataset["images"] = []
    generator.coco_dataset["annotations"] = []
    
    # Generate validation data
    val_results = generator.generate_dataset(args.num_val, "val")
    
    print("\n🎉 Training Data Generation Complete!")
    print(f"Training: {train_results['num_images']} images, {train_results['num_annotations']} annotations")
    print(f"Validation: {val_results['num_images']} images, {val_results['num_annotations']} annotations")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
