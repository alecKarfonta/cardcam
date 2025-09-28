#!/usr/bin/env python3
"""
Quick YOLO OBB Model Test - Tests on just a few images for fast results
"""

import os
import glob
import argparse
from pathlib import Path
from ultralytics import YOLO
import time
import json
import cv2
import numpy as np
from datetime import datetime
import zipfile
import shutil

def parse_yolo_obb_label(label_path, img_width, img_height):
    """Parse YOLO OBB label file and convert to pixel coordinates."""
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 9:  # class + 8 coordinates
                continue
                
            # Skip class ID, get normalized coordinates
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_coords = []
            for i in range(0, 8, 2):
                x = coords[i] * img_width
                y = coords[i + 1] * img_height
                pixel_coords.extend([x, y])
            
            # Reshape to 4x2 array (4 corners, x,y)
            box = np.array(pixel_coords).reshape(4, 2).astype(np.int32)
            boxes.append(box)
    
    return boxes


def draw_obb_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw oriented bounding boxes on image."""
    result = image.copy()
    
    for box in boxes:
        # Draw the oriented bounding box
        cv2.polylines(result, [box], True, color, thickness)
        
        # Draw corner points
        for point in box:
            cv2.circle(result, tuple(point), 3, color, -1)
            
        # Draw center point
        center = np.mean(box, axis=0).astype(np.int32)
        cv2.circle(result, tuple(center), 5, (0, 0, 255), -1)
    
    return result


def create_gold_visualization(gold_labels_dir, gold_images_dir, output_dir, max_images=20):
    """Create visualization of gold labels overlaid on images."""
    print(f"\n🎨 Creating gold labels visualization...")
    
    # Create output directory
    viz_output = Path(output_dir) / "gold_visualizations"
    viz_output.mkdir(parents=True, exist_ok=True)
    
    # Get image files that have corresponding labels
    label_files = list(Path(gold_labels_dir).glob("*.txt"))
    if not label_files:
        print("   ⚠️  No label files found")
        return None
    
    # Find corresponding images
    image_label_pairs = []
    for label_file in label_files[:max_images]:
        image_name = label_file.stem
        image_path = Path(gold_images_dir) / f"{image_name}.jpg"
        if image_path.exists():
            image_label_pairs.append((str(image_path), str(label_file)))
    
    if not image_label_pairs:
        print("   ⚠️  No matching image-label pairs found")
        return None
    
    print(f"   📊 Processing {len(image_label_pairs)} image-label pairs")
    
    # Statistics
    stats = {
        "total_pairs": len(image_label_pairs),
        "total_detections": 0,
        "images_processed": 0
    }
    
    # Process images and create grid
    grid_images = []
    grid_size = (4, 5)  # cols, rows
    
    for i, (img_path, label_path) in enumerate(image_label_pairs):
        image_name = Path(img_path).stem
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        h, w = image.shape[:2]
        
        # Parse labels
        boxes = parse_yolo_obb_label(label_path, w, h)
        stats["total_detections"] += len(boxes)
        
        # Draw bounding boxes
        viz_image = draw_obb_on_image(image, boxes)
        
        # Add text overlay
        text = f"{image_name} ({len(boxes)} detections)"
        cv2.putText(viz_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1)
        
        # Save individual visualization
        individual_output = viz_output / f"individual_{image_name}.jpg"
        cv2.imwrite(str(individual_output), viz_image)
        
        # Add to grid (resize for grid)
        if len(grid_images) < grid_size[0] * grid_size[1]:
            grid_img = cv2.resize(viz_image, (400, 300))
            grid_images.append(grid_img)
        
        stats["images_processed"] += 1
    
    # Create grid visualization
    if grid_images:
        cols, rows = grid_size
        grid_h, grid_w = 300, 400
        
        # Create empty grid
        grid_canvas = np.zeros((rows * grid_h, cols * grid_w, 3), dtype=np.uint8)
        
        for idx, img in enumerate(grid_images):
            if idx >= cols * rows:
                break
                
            row = idx // cols
            col = idx % cols
            
            y1, y2 = row * grid_h, (row + 1) * grid_h
            x1, x2 = col * grid_w, (col + 1) * grid_w
            
            grid_canvas[y1:y2, x1:x2] = img
        
        # Add title to grid
        title_height = 60
        title_canvas = np.zeros((title_height, cols * grid_w, 3), dtype=np.uint8)
        title_text = f"Gold Dataset Labels ({stats['images_processed']} images, {stats['total_detections']} detections)"
        cv2.putText(title_canvas, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        # Combine title and grid
        final_grid = np.vstack([title_canvas, grid_canvas])
        
        # Save grid visualization
        grid_output = viz_output / "gold_labels_grid.jpg"
        cv2.imwrite(str(grid_output), final_grid)
    
    print(f"   ✅ Visualization saved to: {viz_output}")
    print(f"   📊 Processed {stats['images_processed']} images with {stats['total_detections']} total detections")
    
    return stats


def save_coco_annotation(result, image_path, image_id, annotation_id_start, annotation_threshold=0.5):
    """Save predictions in COCO Instances format with polygon segmentation and return annotations list.

    Notes:
    - Each oriented box is exported as a single polygon segmentation with 8 points (x1,y1,...,x4,y4)
    - BBox is axis-aligned [x, y, w, h] as per COCO spec
    - Area is computed via the polygon shoelace formula
    """
    annotations = []

    if result.obb is None or len(result.obb.xyxyxyxy) == 0:
        return annotations

    # Get image dimensions (not strictly required for COCO, but kept for completeness)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    annotation_id = annotation_id_start

    for i in range(len(result.obb.xyxyxyxy)):
        # Get coordinates and confidence
        coords = result.obb.xyxyxyxy[i].cpu().numpy().flatten()
        conf = result.obb.conf[i].cpu().numpy().item()
        cls = int(result.obb.cls[i].cpu().numpy().item())

        # Only save if confidence meets threshold
        if conf >= annotation_threshold:
            # Axis-aligned bbox from polygon
            x_coords = coords[::2]
            y_coords = coords[1::2]

            bbox_x = float(np.min(x_coords))
            bbox_y = float(np.min(y_coords))
            bbox_w = float(np.max(x_coords) - np.min(x_coords))
            bbox_h = float(np.max(y_coords) - np.min(y_coords))

            # Shoelace formula for polygon area
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

            # COCO polygon segmentation is a list of lists
            segmentation = [coords.tolist()]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Single class: "card"
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation,
                "confidence": float(conf)
            }

            annotations.append(annotation)
            annotation_id += 1

    return annotations


def save_yolo_obb_annotation(result, image_path, output_dir, annotation_threshold=0.5):
    """Save predictions in YOLO OBB format, filtering by confidence threshold."""
    if result.obb is None or len(result.obb.xyxyxyxy) == 0:
        return 0
    
    # Get image dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Create output file
    image_name = Path(image_path).stem
    label_path = output_dir / f"{image_name}.txt"
    
    def canonicalize_obb_points(points: np.ndarray) -> np.ndarray:
        """Return 4x2 points ordered clockwise, starting from top-left corner.

        - Sort by angle around centroid (CCW), then reverse for clockwise
        - Rotate so the first point is closest to the axis-aligned top-left of the polygon
        """
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order_ccw = np.argsort(angles)
        ordered = points[order_ccw]
        # Make clockwise
        ordered = ordered[::-1]
        # Find top-left: closest to (min_x, min_y) of ordered points
        min_x = float(np.min(ordered[:, 0]))
        min_y = float(np.min(ordered[:, 1]))
        dists = (ordered[:, 0] - min_x) ** 2 + (ordered[:, 1] - min_y) ** 2
        start_idx = int(np.argmin(dists))
        return np.roll(ordered, -start_idx, axis=0)

    saved_count = 0
    with open(label_path, 'w') as f:
        for i in range(len(result.obb.xyxyxyxy)):
            # Get coordinates and confidence
            coords = result.obb.xyxyxyxy[i].cpu().numpy().flatten()
            conf = result.obb.conf[i].cpu().numpy().item()
            cls = int(result.obb.cls[i].cpu().numpy().item())
            
            # Only save if confidence meets threshold
            if conf >= annotation_threshold:
                # Canonicalize point order (clockwise, start top-left)
                pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                pts = canonicalize_obb_points(pts)

                # Normalize coordinates and clamp to [0,1]
                normalized_coords = []
                for (x, y) in pts:
                    nx = max(0.0, min(1.0, float(x) / float(w)))
                    ny = max(0.0, min(1.0, float(y) / float(h)))
                    normalized_coords.extend([nx, ny])
                
                # Write in YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                coord_str = ' '.join([f"{coord:.6f}" for coord in normalized_coords])
                f.write(f"{cls} {coord_str}\n")
                saved_count += 1
    
    # Remove file if no annotations were saved
    if saved_count == 0:
        label_path.unlink(missing_ok=True)
    
    return saved_count


def build_yolo_obb_annotations_zip(labels_dir: Path, classes: list, zip_path: Path):
    """Create a CVAT-importable Ultralytics OBB annotations zip following CVAT's expected structure."""
    labels = sorted(labels_dir.glob("*.txt"))
    if not labels:
        return None

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Write data.yaml in correct format
        yaml_lines = [
            "names:",
        ]
        for idx, name in enumerate(classes):
            yaml_lines.append(f"  {idx}: {name}")
        yaml_lines.extend([
            "path: .",
            "train: train.txt"
        ])
        yaml_content = "\n".join(yaml_lines) + "\n"
        zf.writestr("data.yaml", yaml_content)

        # Write train.txt with proper paths (matching CVAT export format)
        train_txt_lines = []
        for lbl in labels:
            img_name = lbl.stem + ".jpg"
            train_txt_lines.append(f"data/images/train/{img_name}")
        train_txt_content = "\n".join(train_txt_lines) + "\n"
        zf.writestr("train.txt", train_txt_content)

        # Write label files under labels/train/ (CVAT export structure)
        for lbl in labels:
            arcname = f"labels/train/{lbl.name}"
            zf.write(str(lbl), arcname)

    return zip_path


def build_ultralytics_obb_dataset_zip(images_dir: Path, labels_dir: Path, classes: list, zip_path: Path):
    """Create a YOLO Ultralytics OBB dataset ZIP containing images/, labels/, and data.yaml.

    CVAT's yolo_ultralytics_oriented_boxes importer may require dataset-style zips with a YAML.
    This builder places all images under images/ and their label files under labels/.
    """
    image_files = sorted([p for p in images_dir.glob('*.jpg')] + [p for p in images_dir.glob('*.png')])
    label_files = {p.stem: p for p in labels_dir.glob('*.txt')}
    if not image_files or not label_files:
        return None

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # data.yaml
        yaml_lines = [
            f"nc: {len(classes)}",
            "names:",
        ]
        for name in classes:
            yaml_lines.append(f"  - {name}")
        yaml_content = "\n".join(yaml_lines) + "\n"
        zf.writestr("data.yaml", yaml_content)

        # classes.txt (not strictly required, but helpful)
        zf.writestr("classes.txt", "\n".join(classes) + "\n")

        # images and labels
        for img in image_files:
            zf.write(str(img), arcname=f"images/{img.name}")
            lbl = label_files.get(img.stem)
            if lbl is not None:
                zf.write(str(lbl), arcname=f"labels/{lbl.name}")

    return zip_path

def quick_test(model_path="trading_cards_obb/yolo11n_obb_gpu_batch30/weights/best.pt", 
               num_test_images=5, 
               include_gold=False, 
               num_gold_images=10,
               auto_annotate=False,
               conf_threshold=0.25,
               annotation_threshold=0.5,
               test_images_dir="/home/alec/git/pokemon/data/training_100k/images",
               gold_images_dir="/home/alec/git/pokemon/data/gold/images"):
    print("🚀 Quick YOLO OBB Model Test")
    print("=" * 50)
    
    # Load the best model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    
    # Get test images
    test_images = []
    
    # Add synthetic test images
    if num_test_images > 0:
        synthetic_images = glob.glob(f"{test_images_dir}/test_*.jpg")[:num_test_images]
        test_images.extend(synthetic_images)
        print(f"📊 Added {len(synthetic_images)} synthetic test images from {test_images_dir}")
    
    # Add gold dataset images
    if include_gold:
        gold_images = glob.glob(f"{gold_images_dir}/*.jpg")[:num_gold_images]
        test_images.extend(gold_images)
        print(f"🏆 Added {len(gold_images)} gold dataset images from {gold_images_dir}")
    
    print(f"\n🖼️  Testing on {len(test_images)} images...")
    
    # Setup output directories
    output_dir = Path("outputs/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    auto_annotations_dir = None
    gold_labels_dir = None
    gold_coco_dir = None
    coco_data = None
    
    if auto_annotate:
        auto_annotations_dir = Path("auto-annotations")
        auto_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup gold labels directory to mirror data/yolo_obb structure
        gold_labels_dir = Path("data/gold/labels/test")
        gold_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup gold COCO annotations directory
        gold_coco_dir = Path("data/gold/annotations")
        gold_coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO format data structure
        coco_data = {
            "info": {
                "description": "Gold Dataset Auto-Annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLO OBB Auto-Annotation",
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
        
        print(f"📝 Auto-annotations will be saved to: {auto_annotations_dir}")
        print(f"🏆 Gold labels will be saved to: {gold_labels_dir}")
        print(f"🏆 Gold COCO annotations will be saved to: {gold_coco_dir}")
        print(f"📊 Annotation threshold: {annotation_threshold} (only predictions above this confidence will be saved)")
    
    results_summary = []
    total_inference_time = 0
    
    for i, img_path in enumerate(test_images):
        image_name = os.path.basename(img_path)
        image_type = "🏆 Gold" if "gold" in img_path else "📊 Synthetic"
        print(f"\nTesting image {i+1}: {image_name} ({image_type})")
        
        # Run inference (1088 is multiple of 32, required by YOLO)
        start_time = time.time()
        results = model(img_path, conf=conf_threshold, iou=0.7, imgsz=1088, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        total_inference_time += inference_time
        
        result = results[0]
        
        # Extract results (use OBB for oriented bounding boxes)
        num_detections = len(result.obb.xyxyxyxy) if result.obb is not None else 0
        confidences = result.obb.conf.cpu().numpy().tolist() if result.obb is not None else []
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Save visualization
        result.save(str(output_dir / f"result_{i+1}_{image_name}"))
        
        # Save auto-annotations if requested
        saved_annotations = 0
        saved_gold_annotations = 0
        if auto_annotate:
            # Save to auto-annotations for all images
            if auto_annotations_dir:
                saved_annotations = save_yolo_obb_annotation(result, img_path, auto_annotations_dir, annotation_threshold)
            
            # Save gold images to data/gold/labels/test to mirror yolo_obb structure
            if "gold" in img_path and gold_labels_dir:
                saved_gold_annotations = save_yolo_obb_annotation(result, img_path, gold_labels_dir, annotation_threshold)
                
                # Also add to COCO format data
                if coco_data is not None:
                    # Add image info to COCO data
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    
                    image_info = {
                        "id": i + 1,
                        "width": w,
                        "height": h,
                        "file_name": os.path.basename(img_path),
                        "path": img_path
                    }
                    coco_data["images"].append(image_info)
                    
                    # Add annotations to COCO data
                    current_annotation_id = len(coco_data["annotations"]) + 1
                    new_annotations = save_coco_annotation(result, img_path, i + 1, current_annotation_id, annotation_threshold)
                    coco_data["annotations"].extend(new_annotations)
        
        result_info = {
            'image': image_name,
            'image_path': img_path,
            'image_type': 'gold' if 'gold' in img_path else 'synthetic',
            'detections': num_detections,
            'avg_confidence': avg_confidence,
            'inference_time_ms': inference_time,
            'saved_annotations': saved_annotations,
            'saved_gold_annotations': saved_gold_annotations
        }
        results_summary.append(result_info)
        
        print(f"   Detections: {num_detections}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        print(f"   Inference Time: {inference_time:.1f}ms")
        if auto_annotate:
            if saved_annotations > 0:
                print(f"   📝 Auto-annotations saved: {saved_annotations}")
            if saved_gold_annotations > 0:
                print(f"   🏆 Gold labels saved: {saved_gold_annotations}")
            if saved_annotations == 0 and saved_gold_annotations == 0 and num_detections > 0:
                print(f"   ⚠️  No annotations saved (all below threshold {annotation_threshold})")
    
    # Summary statistics
    avg_inference_time = total_inference_time / len(test_images) if test_images else 0
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    total_detections = sum(r['detections'] for r in results_summary)
    avg_detections = total_detections / len(results_summary) if results_summary else 0
    detection_rate = sum(1 for r in results_summary if r['detections'] > 0) / len(results_summary) if results_summary else 0
    
    # Separate stats by image type
    synthetic_results = [r for r in results_summary if r['image_type'] == 'synthetic']
    gold_results = [r for r in results_summary if r['image_type'] == 'gold']
    
    print("\n" + "=" * 60)
    print("📊 QUICK TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    if auto_annotate:
        print(f"Annotation Threshold: {annotation_threshold}")
    print(f"Total Images Tested: {len(test_images)}")
    
    if synthetic_results:
        synthetic_detections = sum(r['detections'] for r in synthetic_results)
        print(f"\n📊 Synthetic Images ({len(synthetic_results)}):")
        print(f"   Total Detections: {synthetic_detections}")
        print(f"   Avg Detections/Image: {synthetic_detections / len(synthetic_results):.1f}")
        print(f"   Detection Rate: {sum(1 for r in synthetic_results if r['detections'] > 0) / len(synthetic_results):.1%}")
    
    if gold_results:
        gold_detections = sum(r['detections'] for r in gold_results)
        print(f"\n🏆 Gold Images ({len(gold_results)}):")
        print(f"   Total Detections: {gold_detections}")
        print(f"   Avg Detections/Image: {gold_detections / len(gold_results):.1f}")
        print(f"   Detection Rate: {sum(1 for r in gold_results if r['detections'] > 0) / len(gold_results):.1%}")
        print(f"   Avg Confidence: {np.mean([r['avg_confidence'] for r in gold_results if r['avg_confidence'] > 0]):.3f}")
    
    print(f"\n⚡ Performance:")
    print(f"   Avg Inference Time: {avg_inference_time:.1f}ms")
    print(f"   FPS: {fps:.1f}")
    
    print(f"\n📁 Output:")
    print(f"   Visualizations: outputs/quick_test/")
    if auto_annotate:
        # Count files with saved annotations (above threshold)
        annotations_files = len([r for r in results_summary if r.get('saved_annotations', 0) > 0])
        gold_annotations_files = len([r for r in results_summary if r.get('saved_gold_annotations', 0) > 0])
        
        # Count total saved annotations
        total_saved_annotations = sum(r.get('saved_annotations', 0) for r in results_summary)
        total_saved_gold_annotations = sum(r.get('saved_gold_annotations', 0) for r in results_summary)
        
        print(f"   Auto-annotations: auto-annotations/ ({annotations_files} files, {total_saved_annotations} annotations)")
        if gold_annotations_files > 0:
            print(f"   Gold labels: data/gold/labels/test/ ({gold_annotations_files} files, {total_saved_gold_annotations} annotations)")
            print(f"   Gold COCO: data/gold/annotations/gold_annotations.json ({total_saved_gold_annotations} annotations)")
            print(f"   Gold visualizations: outputs/gold_visualizations/")
        
        # Show filtering stats
        total_detections_all = sum(r['detections'] for r in results_summary)
        if total_detections_all > total_saved_annotations + total_saved_gold_annotations:
            filtered_out = total_detections_all - (total_saved_annotations + total_saved_gold_annotations)
            print(f"   📊 Filtered out {filtered_out} low-confidence detections (< {annotation_threshold})")
    
    # Save results
    summary = {
        'model_path': model_path,
        'confidence_threshold': conf_threshold,
        'annotation_threshold': annotation_threshold if auto_annotate else None,
        'test_images': len(test_images),
        'synthetic_images': len(synthetic_results),
        'gold_images': len(gold_results),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'detection_rate': detection_rate,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps,
        'auto_annotate_enabled': auto_annotate,
        'individual_results': results_summary
    }
    
    with open('outputs/quick_test/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: outputs/quick_test/results_summary.json")
    
    # Save COCO format annotations if we have gold data
    coco_file = None
    if auto_annotate and coco_data and len(coco_data["annotations"]) > 0:
        coco_file = gold_coco_dir / "gold_annotations.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"📄 COCO annotations saved to: {coco_file}")
    
    # Create gold labels visualization if we generated any gold labels
    if auto_annotate and gold_labels_dir and gold_annotations_files > 0:
        viz_stats = create_gold_visualization(
            gold_labels_dir=str(gold_labels_dir),
            gold_images_dir=gold_images_dir,
            output_dir=str(output_dir.parent)
        )
        if viz_stats:
            summary['gold_visualization_stats'] = viz_stats

    # Build CVAT-ready exports (annotations only, no images required)
    if auto_annotate:
        export_dir = Path("outputs/cvat_exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1) Ultralytics YOLO-OBB annotations zip from gold labels (if any)
        if gold_labels_dir and gold_annotations_files > 0:
            yolo_obb_zip = export_dir / "yolo_obb_gold_labels.zip"
            built = build_yolo_obb_annotations_zip(gold_labels_dir, ["card"], yolo_obb_zip)
            if built:
                print(f"📦 CVAT Ultralytics OBB annotations export: {built}")

        # 2) COCO annotations copy + zip (polygons)
        if coco_file and coco_file.exists():
            coco_export_json = export_dir / "coco_gold_annotations.json"
            try:
                shutil.copyfile(str(coco_file), str(coco_export_json))
                print(f"📦 CVAT COCO annotations export (JSON): {coco_export_json}")
            except Exception as e:
                print(f"⚠️  Failed to copy COCO annotations for export: {e}")

            # Also package as a ZIP with the JSON at the root as annotations.json
            try:
                coco_zip = export_dir / "coco_gold_annotations.zip"
                with zipfile.ZipFile(coco_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(str(coco_file), arcname="annotations.json")
                print(f"📦 CVAT COCO annotations export (ZIP): {coco_zip}")
            except Exception as e:
                print(f"⚠️  Failed to build COCO ZIP export: {e}")

            # Additionally, package a dataset-style ZIP expected by some CVAT flows:
            # annotations/instances_default.json
            try:
                coco_zip_dataset = export_dir / "coco_instances_dataset.zip"
                with zipfile.ZipFile(coco_zip_dataset, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(str(coco_file), arcname="annotations/instances_default.json")
                print(f"📦 CVAT COCO dataset export (ZIP): {coco_zip_dataset}")
            except Exception as e:
                print(f"⚠️  Failed to build COCO dataset ZIP export: {e}")

        # Note: Skipping full Ultralytics OBB dataset with images to keep uploads small.
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick YOLO OBB Model Test with Gold Dataset Support")
    
    parser.add_argument("--model", type=str, default="src/training/trading_cards_obb/yolo11n_obb_v14/weights/best.pt",
                        help="Path to YOLO OBB model weights")
    parser.add_argument("--num-test", type=int, default=5,
                        help="Number of synthetic test images to use (0 to skip)")
    parser.add_argument("--include-gold", action="store_true", default=True,
                        help="Include gold dataset images in testing")
    parser.add_argument("--num-gold", type=int, default=10,
                        help="Number of gold images to test (when --include-gold is used)")
    parser.add_argument("--auto-annotate", action="store_true",
                        help="Save predictions as YOLO OBB format annotations in auto-annotations/")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--annotation-threshold", type=float, default=0.5,
                        help="Confidence threshold for saving auto-annotations (higher = cleaner training data)")
    parser.add_argument("--test-images-dir", type=str, default="/home/alec/git/pokemon/data/training_100k/images",
                        help="Directory containing synthetic test images")
    parser.add_argument("--gold-images-dir", type=str, default="/home/alec/git/pokemon/data/gold/images",
                        help="Directory containing gold dataset images")
    
    args = parser.parse_args()
    
    quick_test(
        model_path=args.model,
        num_test_images=args.num_test,
        include_gold=args.include_gold,
        num_gold_images=args.num_gold,
        auto_annotate=args.auto_annotate,
        conf_threshold=args.conf,
        annotation_threshold=args.annotation_threshold,
        test_images_dir=args.test_images_dir,
        gold_images_dir=args.gold_images_dir
    )
