#!/usr/bin/env python3
"""
Fix Gold Images - Remove EXIF and Pre-resize

This script processes the gold images to:
1. Remove EXIF metadata that YOLO thinks is corrupt
2. Resize to a reasonable training size
3. Save as clean JPEGs without corruption warnings
"""

import os
import sys
from pathlib import Path
import shutil
import random
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    import cv2
import numpy as np

def fix_gold_images(
    gold_dir: str = "/home/alec/git/pokemon/training_app/data/gold",
    max_size: int = 2048,  # Resize to max 2048 on longest side
    quality: int = 95,
):
    """
    Fix gold images by removing EXIF and resizing.
    
    Args:
        gold_dir: Path to gold data directory
        max_size: Maximum size for longest dimension
        quality: JPEG quality (1-100)
    """
    gold_path = Path(gold_dir)
    images_dir = gold_path / "images"
    backup_dir = gold_path / "images_original_backup"
    
    # Create backup of originals
    if not backup_dir.exists():
        print(f"Creating backup of original images...")
        backup_dir.mkdir(parents=True)
        
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.jpeg")) + \
                     list(images_dir.glob("*.JPG"))
        
        for img_file in image_files:
            backup_file = backup_dir / img_file.name
            if not backup_file.exists():
                shutil.copy2(img_file, backup_file)
        
        print(f"Backed up {len(image_files)} images to {backup_dir}")
    else:
        print(f"Backup already exists at {backup_dir}")
    
    # Process images
    image_files = list(images_dir.glob("*.jpg")) + \
                 list(images_dir.glob("*.jpeg")) + \
                 list(images_dir.glob("*.JPG"))
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for i, img_file in enumerate(image_files, 1):
        try:
            if PIL_AVAILABLE:
                # Use PIL/Pillow
                img = Image.open(img_file)
                
                # Apply EXIF orientation BEFORE resizing
                # This ensures the image is in the same orientation as when labels were created
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass  # If EXIF transpose fails, continue with original
                
                orig_width, orig_height = img.size
                
                # Calculate new size
                if max(orig_width, orig_height) > max_size:
                    if orig_width > orig_height:
                        new_width = max_size
                        new_height = int(orig_height * (max_size / orig_width))
                    else:
                        new_height = max_size
                        new_width = int(orig_width * (max_size / orig_height))
                    
                    # Resize
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized = True
                else:
                    new_width, new_height = orig_width, orig_height
                    resized = False
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save without EXIF data
                img.save(
                    img_file,
                    'JPEG',
                    quality=quality,
                    optimize=True,
                    exif=b''  # Remove all EXIF data
                )
                
            else:
                # Use OpenCV as fallback
                img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  ERROR reading {img_file.name}")
                    continue
                
                # Note: OpenCV ignores EXIF orientation, so we need to handle it
                # For now, recommend using PIL which handles it automatically
                # If you see misaligned boxes, make sure PIL is installed
                
                orig_height, orig_width = img.shape[:2]
                
                # Calculate new size
                if max(orig_width, orig_height) > max_size:
                    if orig_width > orig_height:
                        new_width = max_size
                        new_height = int(orig_height * (max_size / orig_width))
                    else:
                        new_height = max_size
                        new_width = int(orig_width * (max_size / orig_height))
                    
                    # Resize
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    resized = True
                else:
                    new_width, new_height = orig_width, orig_height
                    resized = False
                
                # Save without EXIF
                cv2.imwrite(str(img_file), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if resized:
                print(f"  [{i}/{len(image_files)}] {img_file.name}: {orig_width}x{orig_height} -> {new_width}x{new_height}")
            else:
                print(f"  [{i}/{len(image_files)}] {img_file.name}: {orig_width}x{orig_height} (EXIF removed)")
            
        except Exception as e:
            print(f"  ERROR processing {img_file.name}: {e}")
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"   Original images backed up to: {backup_dir}")
    print(f"   Processed images in: {images_dir}")
    print(f"\nNext steps:")
    print(f"1. Delete merged dataset: rm -rf {gold_path.parent}/merged_dataset")
    print(f"2. Re-prepare gold dataset: python3 scripts/prepare_gold_dataset.py")
    print(f"3. Restart training: ./scripts/start_training.sh")


def create_visualization_examples(
    gold_dir: str = "/home/alec/git/pokemon/training_app/data/gold",
    num_examples: int = 6,
):
    """
    Create visualization examples with labels overlaid on images.
    
    Args:
        gold_dir: Path to gold data directory
        num_examples: Number of example visualizations to create
    """
    gold_path = Path(gold_dir)
    
    # Check if dataset has been prepared (train/val splits exist)
    if not (gold_path / "images_split" / "train").exists():
        print("⚠️  Dataset not yet prepared. Run prepare_gold_dataset.py first.")
        return
    
    train_images_dir = gold_path / "images_split" / "train"
    train_labels_dir = gold_path / "labels" / "train"
    output_dir = gold_path / "visualization_examples"
    output_dir.mkdir(exist_ok=True)
    
    # Get all training images
    image_files = list(train_images_dir.glob("*.jpg")) + \
                 list(train_images_dir.glob("*.jpeg")) + \
                 list(train_images_dir.glob("*.JPG"))
    
    if not image_files:
        print("⚠️  No training images found")
        return
    
    # Select random images
    sample_images = random.sample(image_files, min(num_examples, len(image_files)))
    
    print(f"\n🎨 Creating {len(sample_images)} visualization examples...")
    
    for i, img_file in enumerate(sample_images, 1):
        try:
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                print(f"  [{i}/{len(sample_images)}] Skipping {img_file.name} - no labels")
                continue
            
            # Read image
            if PIL_AVAILABLE:
                img = Image.open(img_file)
                draw = ImageDraw.Draw(img)
                width, height = img.size
            else:
                img = cv2.imread(str(img_file))
                height, width = img.shape[:2]
            
            # Read labels (YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 - normalized)
            with open(label_file, 'r') as f:
                labels = f.readlines()
            
            num_cards = len(labels)
            
            # Draw each bounding box
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 9:  # OBB format has 9 values
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:9]))
                    
                    # Convert normalized coordinates to pixels
                    points = []
                    for j in range(0, 8, 2):
                        x = int(coords[j] * width)
                        y = int(coords[j+1] * height)
                        points.append((x, y))
                    
                    # Draw the oriented bounding box
                    if PIL_AVAILABLE:
                        # Draw polygon
                        draw.polygon(points, outline='lime', width=3)
                        # Draw corner circles
                        for point in points:
                            draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], 
                                       fill='red', outline='red')
                    else:
                        # Draw polygon
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, (0, 255, 0), 3)
                        # Draw corner circles
                        for point in points:
                            cv2.circle(img, point, 5, (0, 0, 255), -1)
            
            # Add text label
            text = f"{num_cards} card{'s' if num_cards != 1 else ''} detected"
            if PIL_AVAILABLE:
                # Try to use a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                # Draw text background
                bbox = draw.textbbox((10, 10), text, font=font)
                draw.rectangle(bbox, fill='black')
                draw.text((10, 10), text, fill='lime', font=font)
            else:
                cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Save visualization
            output_file = output_dir / f"example_{i:02d}_{img_file.name}"
            if PIL_AVAILABLE:
                img.save(output_file, 'JPEG', quality=95)
            else:
                cv2.imwrite(str(output_file), img)
            
            print(f"  [{i}/{len(sample_images)}] Created {output_file.name} ({num_cards} cards)")
            
        except Exception as e:
            print(f"  [{i}/{len(sample_images)}] ERROR: {img_file.name} - {e}")
            continue
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   View these to confirm labels are correctly aligned with images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix gold images by removing EXIF and resizing")
    parser.add_argument("--gold-dir", type=str,
                       default="/home/alec/git/pokemon/training_app/data/gold",
                       help="Path to gold data directory")
    parser.add_argument("--max-size", type=int, default=2048,
                       help="Maximum size for longest dimension (default: 2048)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization examples with labels overlaid")
    parser.add_argument("--num-examples", type=int, default=6,
                       help="Number of visualization examples to create (default: 6)")
    parser.add_argument("--only-visualize", action="store_true",
                       help="Only create visualizations, skip image processing")
    
    args = parser.parse_args()
    
    if not args.only_visualize:
        fix_gold_images(
            gold_dir=args.gold_dir,
            max_size=args.max_size,
            quality=args.quality
        )
    
    if args.visualize or args.only_visualize:
        create_visualization_examples(
            gold_dir=args.gold_dir,
            num_examples=args.num_examples
        )

