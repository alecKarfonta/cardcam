#!/usr/bin/env python3
"""
Quick YOLO OBB Model Test - Tests on just a few images for fast results
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO
import time
import json

def quick_test():
    print("🚀 Quick YOLO OBB Model Test")
    print("=" * 50)
    
    # Load the best model
    model_path = "trading_cards_obb/yolo11n_obb_gpu_batch30/weights/best.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    
    # Get a few test images
    test_images_dir = "/home/alec/git/pokemon/data/training_100k/images"
    test_images = glob.glob(f"{test_images_dir}/test_*.jpg")[:5]  # Just 5 test images
    
    print(f"\n🖼️  Testing on {len(test_images)} images...")
    
    results_summary = []
    total_inference_time = 0
    
    for i, img_path in enumerate(test_images):
        print(f"\nTesting image {i+1}: {os.path.basename(img_path)}")
        
        # Run inference (1088 is multiple of 32, required by YOLO)
        start_time = time.time()
        results = model(img_path, conf=0.25, iou=0.7, imgsz=1088, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        total_inference_time += inference_time
        
        result = results[0]
        
        # Extract results (use OBB for oriented bounding boxes)
        num_detections = len(result.obb.xyxyxyxy) if result.obb is not None else 0
        confidences = result.obb.conf.cpu().numpy().tolist() if result.obb is not None else []
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Save visualization
        output_dir = Path("outputs/quick_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(str(output_dir / f"result_{i+1}_{os.path.basename(img_path)}"))
        
        result_info = {
            'image': os.path.basename(img_path),
            'detections': num_detections,
            'avg_confidence': avg_confidence,
            'inference_time_ms': inference_time
        }
        results_summary.append(result_info)
        
        print(f"   Detections: {num_detections}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        print(f"   Inference Time: {inference_time:.1f}ms")
    
    # Summary statistics
    avg_inference_time = total_inference_time / len(test_images)
    fps = 1000 / avg_inference_time
    total_detections = sum(r['detections'] for r in results_summary)
    avg_detections = total_detections / len(results_summary)
    detection_rate = sum(1 for r in results_summary if r['detections'] > 0) / len(results_summary)
    
    print("\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Images Tested: {len(test_images)}")
    print(f"Total Detections: {total_detections}")
    print(f"Avg Detections/Image: {avg_detections:.1f}")
    print(f"Detection Rate: {detection_rate:.1%}")
    print(f"Avg Inference Time: {avg_inference_time:.1f}ms")
    print(f"FPS: {fps:.1f}")
    print(f"Visualizations saved to: outputs/quick_test/")
    
    # Save results
    summary = {
        'model_path': model_path,
        'test_images': len(test_images),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'detection_rate': detection_rate,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps,
        'individual_results': results_summary
    }
    
    with open('outputs/quick_test/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Detailed results saved to: outputs/quick_test/results_summary.json")
    
    return summary

if __name__ == "__main__":
    quick_test()
