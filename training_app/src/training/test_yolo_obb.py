#!/usr/bin/env python3
"""
YOLO OBB Model Testing Script

This script tests the trained YOLO OBB model on various test scenarios:
1. Test set evaluation
2. Individual image inference
3. Performance metrics analysis
4. Visualization of predictions
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


def load_model(model_path: str) -> YOLO:
    """Load the trained YOLO OBB model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully!")
    return model


def run_test_evaluation(model: YOLO, dataset_config: str, save_dir: str) -> Dict[str, Any]:
    """Run evaluation on the test dataset."""
    print("\n🧪 Running test set evaluation...")
    
    # Run validation on test set
    results = model.val(
        data=dataset_config,
        split='test',
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True,
        project=save_dir,
        name='test_evaluation'
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
        'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
        'f1_score': float(results.box.f1) if hasattr(results.box, 'f1') else 0.0,
    }
    
    print(f"📊 Test Results:")
    print(f"   mAP@0.5: {metrics['mAP50']:.3f}")
    print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    return metrics


def test_individual_images(model: YOLO, test_images: List[str], save_dir: str) -> List[Dict]:
    """Test model on individual images and save visualizations."""
    print(f"\n🖼️  Testing on {len(test_images)} individual images...")
    
    results_list = []
    output_dir = Path(save_dir) / "individual_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        print(f"Testing image {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # Run inference
        results = model(img_path, conf=0.25, iou=0.7, verbose=False)
        
        # Process results
        result = results[0]
        image_results = {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'detections': len(result.boxes) if result.boxes is not None else 0,
            'confidences': result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else [],
            'inference_time': result.speed['inference'] if hasattr(result, 'speed') else 0.0
        }
        
        # Save visualization
        output_path = output_dir / f"test_{i+1}_{os.path.basename(img_path)}"
        result.save(str(output_path))
        
        results_list.append(image_results)
        
        print(f"   Detections: {image_results['detections']}")
        print(f"   Avg Confidence: {np.mean(image_results['confidences']):.3f}" if image_results['confidences'] else "   No detections")
        print(f"   Inference Time: {image_results['inference_time']:.1f}ms")
    
    return results_list


def benchmark_performance(model: YOLO, test_images: List[str]) -> Dict[str, float]:
    """Benchmark model performance (speed, memory usage)."""
    print(f"\n⚡ Benchmarking performance on {len(test_images)} images...")
    
    inference_times = []
    preprocess_times = []
    postprocess_times = []
    
    # Warm up
    if test_images:
        model(test_images[0], verbose=False)
    
    for img_path in test_images[:10]:  # Test on first 10 images
        if not os.path.exists(img_path):
            continue
            
        results = model(img_path, verbose=False)
        result = results[0]
        
        if hasattr(result, 'speed'):
            preprocess_times.append(result.speed['preprocess'])
            inference_times.append(result.speed['inference'])
            postprocess_times.append(result.speed['postprocess'])
    
    benchmark_results = {
        'avg_preprocess_time': np.mean(preprocess_times) if preprocess_times else 0.0,
        'avg_inference_time': np.mean(inference_times) if inference_times else 0.0,
        'avg_postprocess_time': np.mean(postprocess_times) if postprocess_times else 0.0,
        'total_avg_time': np.mean([p + i + post for p, i, post in zip(preprocess_times, inference_times, postprocess_times)]) if inference_times else 0.0,
        'fps': 1000.0 / np.mean([p + i + post for p, i, post in zip(preprocess_times, inference_times, postprocess_times)]) if inference_times else 0.0
    }
    
    print(f"📈 Performance Metrics:")
    print(f"   Avg Preprocess: {benchmark_results['avg_preprocess_time']:.1f}ms")
    print(f"   Avg Inference: {benchmark_results['avg_inference_time']:.1f}ms")
    print(f"   Avg Postprocess: {benchmark_results['avg_postprocess_time']:.1f}ms")
    print(f"   Total Avg Time: {benchmark_results['total_avg_time']:.1f}ms")
    print(f"   FPS: {benchmark_results['fps']:.1f}")
    
    return benchmark_results


def save_test_report(metrics: Dict, individual_results: List[Dict], benchmark: Dict, save_path: str):
    """Save comprehensive test report."""
    report = {
        'test_date': datetime.now().isoformat(),
        'model_metrics': metrics,
        'individual_test_results': individual_results,
        'performance_benchmark': benchmark,
        'summary': {
            'total_test_images': len(individual_results),
            'avg_detections_per_image': np.mean([r['detections'] for r in individual_results]) if individual_results else 0,
            'avg_confidence': np.mean([np.mean(r['confidences']) for r in individual_results if r['confidences']]) if individual_results else 0,
            'images_with_detections': sum(1 for r in individual_results if r['detections'] > 0),
            'detection_rate': sum(1 for r in individual_results if r['detections'] > 0) / len(individual_results) if individual_results else 0
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {save_path}")
    return report


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trained YOLO OBB model")
    
    parser.add_argument("--model", type=str, default="training/trading_cards_obb/yolo11n_obb_v14/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="configs/yolo_obb_dataset.yaml", help="Dataset config file")
    parser.add_argument("--test-images", type=str, nargs="+", help="Specific test images")
    parser.add_argument("--output", type=str, default="outputs/model_testing", help="Output directory")
    parser.add_argument("--num-test-images", type=int, default=20, help="Number of random test images to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 YOLO OBB Model Testing")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    model = load_model(args.model)
    
    # Run test set evaluation
    test_metrics = run_test_evaluation(model, args.data, str(output_dir))
    
    # Get test images
    if args.test_images:
        test_images = args.test_images
    else:
        # Get random test images from dataset
        test_images_dir = Path("/home/alec/git/pokemon/data/training_100k/images")
        all_images = list(test_images_dir.glob("test_*.jpg"))
        test_images = [str(img) for img in all_images[:args.num_test_images]]
    
    # Test individual images
    individual_results = test_individual_images(model, test_images, str(output_dir))
    
    # Benchmark performance
    benchmark_results = benchmark_performance(model, test_images)
    
    # Save comprehensive report
    report_path = output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = save_test_report(test_metrics, individual_results, benchmark_results, str(report_path))
    
    print("\n" + "=" * 60)
    print("✅ Testing completed successfully!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📄 Detailed report: {report_path}")
    print("=" * 60)
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    print(f"   Model: {args.model}")
    print(f"   Test mAP@0.5: {test_metrics['mAP50']:.3f}")
    print(f"   Average FPS: {benchmark_results['fps']:.1f}")
    print(f"   Detection Rate: {report['summary']['detection_rate']:.1%}")
    print(f"   Avg Detections/Image: {report['summary']['avg_detections_per_image']:.1f}")


if __name__ == "__main__":
    main()
