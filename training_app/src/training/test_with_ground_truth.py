#!/usr/bin/env python3
"""
YOLO OBB Model Testing with Ground Truth Comparison

This script tests the trained YOLO OBB model and compares predictions with actual labels:
1. Load ground truth labels from YOLO OBB format
2. Run inference on test images
3. Compare predictions vs ground truth
4. Calculate detailed metrics per image
5. Visualize predictions vs ground truth
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from datetime import datetime


def load_yolo_obb_label(label_path: str) -> List[Dict]:
    """Load YOLO OBB format label file."""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:  # class + 8 coordinates
                label = {
                    'class': int(parts[0]),
                    'coordinates': [float(x) for x in parts[1:9]],  # 8 coordinates for OBB
                    'confidence': 1.0  # Ground truth has confidence 1.0
                }
                labels.append(label)
    return labels


def calculate_obb_iou(pred_coords: List[float], gt_coords: List[float]) -> float:
    """
    Calculate IoU between two oriented bounding boxes.
    Coordinates are in format [x1, y1, x2, y2, x3, y3, x4, y4] (normalized)
    """
    # For simplicity, we'll use a basic overlap calculation
    # In production, you'd want a more sophisticated OBB IoU calculation
    
    # Convert to numpy arrays and reshape to 4x2 (4 points, 2 coordinates each)
    pred_points = np.array(pred_coords).reshape(4, 2)
    gt_points = np.array(gt_coords).reshape(4, 2)
    
    # Calculate bounding rectangles for approximation
    pred_min = pred_points.min(axis=0)
    pred_max = pred_points.max(axis=0)
    gt_min = gt_points.min(axis=0)
    gt_max = gt_points.max(axis=0)
    
    # Calculate intersection
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    
    if (inter_max > inter_min).all():
        inter_area = np.prod(inter_max - inter_min)
    else:
        inter_area = 0.0
    
    # Calculate union
    pred_area = np.prod(pred_max - pred_min)
    gt_area = np.prod(gt_max - gt_min)
    union_area = pred_area + gt_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_ground_truth(predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match predictions to ground truth labels using IoU threshold.
    Returns: (matched_pairs, unmatched_predictions, unmatched_ground_truth)
    """
    matched_pairs = []
    unmatched_predictions = predictions.copy()
    unmatched_ground_truth = ground_truth.copy()
    
    # Calculate IoU matrix
    for i, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt in unmatched_ground_truth:  # Only match unmatched GT
                iou = calculate_obb_iou(pred['coordinates'], gt['coordinates'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_gt_idx >= 0:
            matched_pairs.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': float(best_iou)
            })
            if pred in unmatched_predictions:
                unmatched_predictions.remove(pred)
            if ground_truth[best_gt_idx] in unmatched_ground_truth:
                unmatched_ground_truth.remove(ground_truth[best_gt_idx])
    
    return matched_pairs, unmatched_predictions, unmatched_ground_truth


def test_image_with_ground_truth(model: YOLO, image_path: str, label_path: str, conf_threshold: float = 0.25) -> Dict:
    """Test a single image and compare with ground truth."""
    
    # Load ground truth
    ground_truth = load_yolo_obb_label(label_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold, iou=0.7, imgsz=1088, verbose=False)
    result = results[0]
    
    # Extract predictions
    predictions = []
    if result.obb is not None and len(result.obb.xyxyxyxy) > 0:
        for i in range(len(result.obb.xyxyxyxy)):
            # Convert from xyxyxyxy format to normalized coordinates
            coords = result.obb.xyxyxyxy[i].cpu().numpy().flatten()
            conf = result.obb.conf[i].cpu().numpy().item()
            cls = result.obb.cls[i].cpu().numpy().item()
            
            # Normalize coordinates by image size
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            normalized_coords = []
            for j in range(0, 8, 2):
                normalized_coords.extend([coords[j]/w, coords[j+1]/h])
            
            predictions.append({
                'class': int(cls),
                'coordinates': [float(x) for x in normalized_coords],
                'confidence': float(conf)
            })
    
    # Match predictions to ground truth
    matched_pairs, false_positives, false_negatives = match_predictions_to_ground_truth(predictions, ground_truth)
    
    # Calculate metrics
    tp = len(matched_pairs)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'ground_truth_count': len(ground_truth),
        'prediction_count': len(predictions),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'matched_pairs': matched_pairs,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_confidence': np.mean([p['confidence'] for p in predictions]) if predictions else 0.0,
        'avg_matched_iou': np.mean([pair['iou'] for pair in matched_pairs]) if matched_pairs else 0.0
    }


def create_comparison_visualization(image_path: str, result_data: Dict, output_path: str):
    """Create a visualization comparing predictions vs ground truth."""
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground Truth visualization
    ax1.imshow(img)
    ax1.set_title(f"Ground Truth ({result_data['ground_truth_count']} objects)")
    ax1.axis('off')
    
    # Draw ground truth boxes in green
    for fn in result_data['false_negatives']:
        coords = np.array(fn['coordinates']).reshape(4, 2)
        coords[:, 0] *= w  # denormalize x
        coords[:, 1] *= h  # denormalize y
        
        # Draw polygon
        polygon = plt.Polygon(coords, fill=False, edgecolor='green', linewidth=2, label='Ground Truth')
        ax1.add_patch(polygon)
    
    # Draw matched ground truth in blue
    for pair in result_data['matched_pairs']:
        coords = np.array(pair['ground_truth']['coordinates']).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        
        polygon = plt.Polygon(coords, fill=False, edgecolor='blue', linewidth=2, label='Matched GT')
        ax1.add_patch(polygon)
    
    # Predictions visualization
    ax2.imshow(img)
    ax2.set_title(f"Predictions ({result_data['prediction_count']} objects)")
    ax2.axis('off')
    
    # Draw false positives in red
    for fp in result_data['false_positives']:
        coords = np.array(fp['coordinates']).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        
        polygon = plt.Polygon(coords, fill=False, edgecolor='red', linewidth=2, label='False Positive')
        ax2.add_patch(polygon)
        
        # Add confidence text
        center = coords.mean(axis=0)
        ax2.text(center[0], center[1], f"{fp['confidence']:.2f}", 
                color='red', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Draw matched predictions in blue
    for pair in result_data['matched_pairs']:
        coords = np.array(pair['prediction']['coordinates']).reshape(4, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        
        polygon = plt.Polygon(coords, fill=False, edgecolor='blue', linewidth=2, label='True Positive')
        ax2.add_patch(polygon)
        
        # Add confidence and IoU text
        center = coords.mean(axis=0)
        ax2.text(center[0], center[1], f"{pair['prediction']['confidence']:.2f}\nIoU:{pair['iou']:.2f}", 
                color='blue', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Add metrics text
    metrics_text = f"Precision: {result_data['precision']:.3f}\nRecall: {result_data['recall']:.3f}\nF1: {result_data['f1_score']:.3f}"
    fig.suptitle(f"{result_data['image_name']} - {metrics_text}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main testing function with ground truth comparison."""
    parser = argparse.ArgumentParser(description="Test YOLO OBB model with ground truth comparison")
    
    parser.add_argument("--model", type=str, default="trading_cards_obb/yolo11n_obb_gpu_batch30/weights/best.pt", help="Path to model weights")
    parser.add_argument("--images-dir", type=str, default="data/training_100k/images", help="Test images directory")
    parser.add_argument("--labels-dir", type=str, default="data/yolo_obb/labels/test", help="Test labels directory")
    parser.add_argument("--output", type=str, default="outputs/ground_truth_comparison", help="Output directory")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images to analyze")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for predictions")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 YOLO OBB Model Testing with Ground Truth Comparison")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("✅ Model loaded successfully!")
    
    # Get test images
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    
    test_images = list(images_dir.glob("test_*.jpg"))[:args.num_images]
    
    print(f"\n🖼️  Testing on {len(test_images)} images...")
    
    all_results = []
    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    
    for i, image_path in enumerate(test_images):
        print(f"\nProcessing image {i+1}/{len(test_images)}: {image_path.name}")
        
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        # Test image with ground truth comparison
        result = test_image_with_ground_truth(model, str(image_path), str(label_path), args.conf_threshold)
        all_results.append(result)
        
        # Create visualization
        viz_path = visualizations_dir / f"{image_path.stem}_comparison.png"
        create_comparison_visualization(str(image_path), result, str(viz_path))
        
        # Print results
        print(f"   Ground Truth: {result['ground_truth_count']} objects")
        print(f"   Predictions: {result['prediction_count']} objects")
        print(f"   True Positives: {result['true_positives']}")
        print(f"   False Positives: {len(result['false_positives'])}")
        print(f"   False Negatives: {len(result['false_negatives'])}")
        print(f"   Precision: {result['precision']:.3f}")
        print(f"   Recall: {result['recall']:.3f}")
        print(f"   F1 Score: {result['f1_score']:.3f}")
        if result['avg_matched_iou'] > 0:
            print(f"   Avg IoU: {result['avg_matched_iou']:.3f}")
    
    # Calculate overall metrics
    total_tp = sum(r['true_positives'] for r in all_results)
    total_fp = sum(len(r['false_positives']) for r in all_results)
    total_fn = sum(len(r['false_negatives']) for r in all_results)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Save detailed report
    report = {
        'test_date': datetime.now().isoformat(),
        'model_path': args.model,
        'confidence_threshold': args.conf_threshold,
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn
        },
        'per_image_results': all_results,
        'summary': {
            'total_images': len(all_results),
            'avg_precision': np.mean([r['precision'] for r in all_results]),
            'avg_recall': np.mean([r['recall'] for r in all_results]),
            'avg_f1': np.mean([r['f1_score'] for r in all_results]),
            'avg_ground_truth_per_image': np.mean([r['ground_truth_count'] for r in all_results]),
            'avg_predictions_per_image': np.mean([r['prediction_count'] for r in all_results])
        }
    }
    
    report_path = output_dir / f"ground_truth_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print(f"Overall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"\nAverage per image:")
    print(f"  Precision: {report['summary']['avg_precision']:.3f}")
    print(f"  Recall: {report['summary']['avg_recall']:.3f}")
    print(f"  F1 Score: {report['summary']['avg_f1']:.3f}")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📄 Detailed report: {report_path}")
    print(f"🖼️  Visualizations: {visualizations_dir}")


if __name__ == "__main__":
    main()
