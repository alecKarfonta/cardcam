#!/usr/bin/env python3
"""
Simple script to test YOLO OBB model on MTG image
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import time
import cv2

def test_mtg_image():
    print("🚀 Testing YOLO OBB Model on MTG Image")
    print("=" * 50)
    
    # Model and image paths
    model_path = "src/training/trading_cards_obb/yolo11n_obb_v16/weights/best.pt"
    image_path = "data/mtg.png"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
        
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"📁 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    
    # Load the model (force CPU to avoid CUDA compatibility issues)
    try:
        print("\n🔄 Loading model...")
        model = YOLO(model_path)
        model.to('cpu')  # Force CPU inference
        print("✅ Model loaded successfully (using CPU)!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Setup output directory
    output_dir = Path("outputs/mtg_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\n🔍 Running inference on {image_path}...")
    try:
        start_time = time.time()
        results = model(image_path, conf=0.25, iou=0.7, imgsz=1088, verbose=False, device='cpu')
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = results[0]
        
        # Extract results
        num_detections = len(result.obb.xyxyxyxy) if result.obb is not None else 0
        confidences = result.obb.conf.cpu().numpy().tolist() if result.obb is not None else []
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        print(f"✅ Inference completed!")
        print(f"   Detections: {num_detections}")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        print(f"   Inference Time: {inference_time:.1f}ms")
        
        if confidences:
            print(f"   Individual confidences: {[f'{c:.3f}' for c in confidences]}")
        
        # Save visualization
        output_path = output_dir / "mtg_detections.jpg"
        result.save(str(output_path))
        print(f"\n📁 Visualization saved to: {output_path}")
        
        # Also save a copy with timestamp
        timestamp_path = output_dir / f"mtg_detections_{int(time.time())}.jpg"
        result.save(str(timestamp_path))
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mtg_image()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
