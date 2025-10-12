#!/usr/bin/env python3
"""
Debug test to understand why the model isn't detecting anything
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def debug_test():
    print("🔍 Debug Test - Investigating Detection Issues")
    print("=" * 60)
    
    # Load model
    model_path = "trading_cards_obb/yolo11n_obb_gpu_rtx5090/weights/best.pt"
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Test on training images (these should definitely work)
    train_images = glob.glob("/home/alec/git/pokemon/data/training_100k/images/train_*.jpg")[:3]
    test_images = glob.glob("/home/alec/git/pokemon/data/training_100k/images/test_*.jpg")[:3]
    
    print(f"\n📊 Testing on TRAINING images (should have detections):")
    test_images_set(model, train_images, "train")
    
    print(f"\n📊 Testing on TEST images:")
    test_images_set(model, test_images, "test")
    
    # Check model info
    print(f"\n🤖 Model Information:")
    print(f"   Model type: {type(model.model)}")
    print(f"   Task: {model.task}")
    print(f"   Device: {model.device}")
    
    # Test with different confidence thresholds
    print(f"\n🎯 Testing different confidence thresholds on first training image:")
    if train_images:
        test_confidence_thresholds(model, train_images[0])

def test_images_set(model, images, set_name):
    for i, img_path in enumerate(images):
        print(f"\n   {set_name.upper()} Image {i+1}: {os.path.basename(img_path)}")
        
        # Test with very low confidence
        results = model(img_path, conf=0.001, iou=0.7, verbose=False)
        result = results[0]
        
        num_detections = len(result.boxes) if result.boxes is not None else 0
        print(f"      Detections (conf=0.001): {num_detections}")
        
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            print(f"      Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
            
            # Save visualization
            output_dir = Path("outputs/debug_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            result.save(str(output_dir / f"debug_{set_name}_{i+1}_{os.path.basename(img_path)}"))
        
        # Check if image loads correctly
        img = cv2.imread(img_path)
        if img is not None:
            print(f"      Image shape: {img.shape}")
        else:
            print(f"      ❌ Could not load image!")

def test_confidence_thresholds(model, img_path):
    thresholds = [0.001, 0.01, 0.1, 0.25, 0.5]
    
    for conf in thresholds:
        results = model(img_path, conf=conf, iou=0.7, verbose=False)
        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0
        print(f"      conf={conf}: {num_detections} detections")

if __name__ == "__main__":
    debug_test()
