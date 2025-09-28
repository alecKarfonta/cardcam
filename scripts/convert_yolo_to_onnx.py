#!/usr/bin/env python3
"""
Convert YOLO OBB model to ONNX format for web deployment
Includes inference testing and comparison between PT and ONNX models
"""

import os
import sys
import torch
import cv2
import numpy as np
import json
import time
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

def letterbox_resize(image, size=1088):
    """Letterbox resize image to square size while maintaining aspect ratio"""
    h, w = image.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    
    # Create canvas and resize image
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # Gray padding
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return canvas, r, (pad_x, pad_y)

def obb_to_polygon(cx, cy, w, h, angle):
    """Convert oriented bounding box to polygon corners"""
    # Convert angle to radians if needed
    if abs(angle) > np.pi:
        angle = np.radians(angle)
    
    # Calculate half dimensions
    hw, hh = w / 2, h / 2
    
    # Corner points relative to center (before rotation)
    corners = np.array([
        [-hw, -hh],  # top-left
        [hw, -hh],   # top-right
        [hw, hh],    # bottom-right
        [-hw, hh]    # bottom-left
    ])
    
    # Rotation matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate corners
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to center position
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy
    
    return rotated_corners

def draw_obb_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Draw oriented bounding box detections on image"""
    result = image.copy()
    
    if detections is None or len(detections) == 0:
        return result
    
    # Handle different detection formats
    if isinstance(detections, list) and len(detections) > 0:
        # Check if it's our custom detection format (list of dicts)
        if isinstance(detections[0], dict) and 'cx' in detections[0]:
            # Our custom ONNX detection format
            for detection in detections:
                cx, cy = detection['cx'], detection['cy']
                w, h = detection['w'], detection['h']
                angle = detection['angle']
                confidence = detection['confidence']
                
                if confidence > 0.25:  # Confidence threshold
                    # Get polygon corners
                    corners = obb_to_polygon(cx, cy, w, h, angle)
                    box = corners.astype(np.int32)
                    
                    # Draw the oriented bounding box
                    cv2.polylines(result, [box], True, color, thickness)
                    
                    # Draw corner points
                    for point in box:
                        cv2.circle(result, tuple(point), 3, color, -1)
                    
                    # Draw confidence score
                    center = np.array([cx, cy]).astype(np.int32)
                    cv2.putText(result, f'{confidence:.2f}', 
                              tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 255), 1)
            return result
    
    # Handle PyTorch YOLO results
    for detection in detections:
        if hasattr(detection, 'obb') and detection.obb is not None:
            # YOLO OBB format - get the rotated box points
            obb_data = detection.obb
            if hasattr(obb_data, 'xyxyxyxy'):
                # Get the 4 corner points
                points = obb_data.xyxyxyxy.cpu().numpy()
                conf = obb_data.conf.cpu().numpy()
                
                for i, (box_points, confidence) in enumerate(zip(points, conf)):
                    if confidence > 0.25:  # Confidence threshold
                        # Reshape to 4x2 array (4 corners, x,y)
                        box = box_points.reshape(4, 2).astype(np.int32)
                        
                        # Draw the oriented bounding box
                        cv2.polylines(result, [box], True, color, thickness)
                        
                        # Draw corner points
                        for point in box:
                            cv2.circle(result, tuple(point), 3, color, -1)
                        
                        # Draw confidence score
                        center = np.mean(box, axis=0).astype(np.int32)
                        cv2.putText(result, f'{confidence:.2f}', 
                                  tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (255, 255, 255), 1)
        elif hasattr(detection, 'boxes') and detection.boxes is not None:
            # Regular bounding boxes fallback
            boxes = detection.boxes
            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                
                for box, confidence in zip(xyxy, conf):
                    if confidence > 0.25:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(result, f'{confidence:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (255, 255, 255), 1)
    
    return result


def run_pt_inference(model_path, image_path, conf_threshold=0.25):
    """Run inference using PyTorch YOLO model"""
    print(f"Running PT inference...")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference (using same parameters as ONNX for comparison)
    start_time = time.time()
    results = model(image_path, conf=conf_threshold, iou=0.7, verbose=False)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    result = results[0]
    num_detections = 0
    confidences = []
    
    if hasattr(result, 'obb') and result.obb is not None:
        num_detections = len(result.obb)
        confidences = result.obb.conf.cpu().numpy().tolist()
    elif hasattr(result, 'boxes') and result.boxes is not None:
        num_detections = len(result.boxes)
        confidences = result.boxes.conf.cpu().numpy().tolist()
    
    return {
        'detections': num_detections,
        'confidences': confidences,
        'inference_time_ms': inference_time,
        'results': results
    }

def run_onnx_inference(onnx_path, image_path, conf_threshold=0.25, input_size=1088):
    """Run inference using ONNX model with built-in NMS"""
    print(f"Running ONNX inference (with built-in NMS)...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_h, original_w = image.shape[:2]
    
    # Preprocess image (letterbox resize)
    letterboxed, ratio, (pad_x, pad_y) = letterbox_resize(image, input_size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    input_tensor = rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    print(f"  ONNX output shapes: {[output.shape for output in outputs]}")
    
    detections = []
    confidences = []
    
    # Parse NMS output format [1, 300, 6] or [1, 300, 7] 
    # With NMS=True: [x1, y1, x2, y2, conf, class] (normalized coordinates)
    if len(outputs) >= 1:
        output = outputs[0]  # Main output
        
        if len(output.shape) == 3 and output.shape[0] == 1:
            # Remove batch dimension: [1, 300, 6/7] -> [300, 6/7]
            output = output[0]
            
            for i in range(output.shape[0]):
                detection = output[i]
                
                # Skip empty detections (all zeros)
                if np.all(detection == 0):
                    continue
                
                if len(detection) >= 6:
                    # Based on debug output, format appears to be: [cx, cy, w, h, conf, class, angle]
                    cx, cy, w, h, confidence, class_id = detection[:6]
                    angle = detection[6] if len(detection) > 6 else 0
                    
                    if confidence > conf_threshold:
                        # Coordinates are already in pixel space (1088x1088 input size)
                        # No need to denormalize
                        
                        # Scale back to original image size
                        scaled_cx = (cx - pad_x) / ratio
                        scaled_cy = (cy - pad_y) / ratio
                        scaled_w = w / ratio
                        scaled_h = h / ratio
                        
                        if (scaled_w > 0 and scaled_h > 0):
                            detections.append({
                                'cx': scaled_cx,
                                'cy': scaled_cy,
                                'w': scaled_w,
                                'h': scaled_h,
                                'angle': float(angle),  # Include rotation angle
                                'confidence': float(confidence),
                                'class': int(class_id)
                            })
                            confidences.append(float(confidence))
    
    return {
        'detections': len(detections),
        'confidences': confidences,
        'inference_time_ms': inference_time,
        'output_shape': [output.shape for output in outputs],
        'processed_detections': detections,
        'nms_built_in': True
    }

def compare_models(pt_results, onnx_results):
    """Compare PT and ONNX model results"""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    print(f"PT Model:")
    print(f"  Detections: {pt_results['detections']}")
    print(f"  Avg Confidence: {np.mean(pt_results['confidences']):.3f}" if pt_results['confidences'] else "  No detections")
    print(f"  Inference Time: {pt_results['inference_time_ms']:.1f}ms")
    
    print(f"\nONNX Model:")
    print(f"  Detections (after NMS): {onnx_results['detections']}")
    if 'raw_detections_count' in onnx_results:
        print(f"  Raw detections (before NMS): {onnx_results['raw_detections_count']}")
    print(f"  Avg Confidence: {np.mean(onnx_results['confidences']):.3f}" if onnx_results['confidences'] else "  No detections")
    print(f"  Inference Time: {onnx_results['inference_time_ms']:.1f}ms")
    print(f"  Output Shape: {onnx_results['output_shape']}")
    
    # Calculate differences
    detection_diff = abs(pt_results['detections'] - onnx_results['detections'])
    time_diff = abs(pt_results['inference_time_ms'] - onnx_results['inference_time_ms'])
    
    print(f"\nDifferences:")
    print(f"  Detection Count Diff: {detection_diff}")
    print(f"  Inference Time Diff: {time_diff:.1f}ms")
    
    if pt_results['confidences'] and onnx_results['confidences']:
        conf_diff = abs(np.mean(pt_results['confidences']) - np.mean(onnx_results['confidences']))
        print(f"  Avg Confidence Diff: {conf_diff:.3f}")
    
    print(f"\n✅ Note: ONNX model now includes proper NMS post-processing.")
    print(f"   Results should be comparable to PyTorch model.")
    
    return {
        'detection_diff': detection_diff,
        'time_diff': time_diff,
        'pt_faster': pt_results['inference_time_ms'] < onnx_results['inference_time_ms']
    }

def test_inference_on_image(model_path, onnx_path, image_path, output_dir):
    """Test both models on the given image and create visualizations"""
    print(f"\n🧪 Testing inference on: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return None
    
    # Create output directory for test results
    test_output_dir = os.path.join(output_dir, "inference_test")
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # Run PT inference
        pt_results = run_pt_inference(model_path, image_path)
        
        # Run ONNX inference (if ONNX file exists)
        onnx_results = None
        if os.path.exists(onnx_path):
            onnx_results = run_onnx_inference(onnx_path, image_path)
        
        # Load original image for visualization
        original_image = cv2.imread(image_path)
        
        # Create PT visualization
        pt_viz = draw_obb_detections(original_image, pt_results['results'])
        pt_output_path = os.path.join(test_output_dir, f"pt_detections_{os.path.basename(image_path)}")
        cv2.imwrite(pt_output_path, pt_viz)
        print(f"📊 PT visualization saved: {pt_output_path}")
        
        # Create ONNX visualization with actual detections
        if onnx_results and 'processed_detections' in onnx_results:
            onnx_viz = draw_obb_detections(original_image, onnx_results['processed_detections'], 
                                         color=(0, 0, 255), thickness=2)  # Red color for ONNX
            
            # Add text overlay showing ONNX results
            cv2.putText(onnx_viz, f"ONNX: {onnx_results['detections']} detections", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(onnx_viz, f"Time: {onnx_results['inference_time_ms']:.1f}ms", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            onnx_output_path = os.path.join(test_output_dir, f"onnx_detections_{os.path.basename(image_path)}")
            cv2.imwrite(onnx_output_path, onnx_viz)
            print(f"📊 ONNX visualization saved: {onnx_output_path}")
        
        # Compare results
        if onnx_results:
            comparison = compare_models(pt_results, onnx_results)
            
            # Save comparison results (convert numpy types to Python types for JSON serialization)
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            comparison_data = {
                'image_path': image_path,
                'pt_results': convert_numpy_types({k: v for k, v in pt_results.items() if k != 'results'}),
                'onnx_results': convert_numpy_types({k: v for k, v in onnx_results.items() if k != 'raw_output'}),
                'comparison': convert_numpy_types(comparison)
            }
            
            comparison_path = os.path.join(test_output_dir, f"comparison_{os.path.basename(image_path)}.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            print(f"📋 Comparison data saved: {comparison_path}")
            
            return comparison_data
        else:
            print("⚠️  ONNX model not found, skipping comparison")
            return {'pt_results': {k: v for k, v in pt_results.items() if k != 'results'}}
            
    except Exception as e:
        print(f"❌ Error during inference testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_yolo_to_onnx(model_path, output_dir, input_size=1088):
    """
    Convert YOLO model to ONNX format
    
    Args:
        model_path: Path to the .pt model file
        output_dir: Directory to save the ONNX model
        input_size: Input image size (default: 640)
    """
    
    print(f"Loading YOLO model from: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to ONNX
    onnx_path = os.path.join(output_dir, "trading_card_detector.onnx")
    
    print(f"Converting to ONNX format...")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Output path: {onnx_path}")
    
    try:
        # Export to ONNX with NMS (working configuration)
        success = model.export(
            format="onnx",
            imgsz=1088,
            optimize=True,
            half=False,  # Use FP32 for better web compatibility
            dynamic=False,  # Fixed input size for web deployment
            simplify=True,  # Simplify the model
            nms=True,  # Enable NMS for proper object detection
            opset=16,  # ONNX opset version (16 works with NMS)
        )
        
        if success:
            # Move the exported file to our desired location
            exported_path = str(model_path).replace('.pt', '.onnx')
            if os.path.exists(exported_path):
                import shutil
                shutil.move(exported_path, onnx_path)
                print(f"✅ Model successfully converted to: {onnx_path}")
                
                # Print model info
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                print(f"📊 Model size: {file_size:.2f} MB")
                
                return onnx_path
            else:
                print(f"❌ Expected ONNX file not found at: {exported_path}")
                return None
        else:
            print("❌ Export failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

def main():
    # Paths
    model_path = "/home/alec/git/pokemon/src/training/trading_cards_obb/yolo11n_obb_v16/weights/best.pt"
    output_dir = "/home/alec/git/pokemon/frontend/public/models"
    test_image_path = "/home/alec/git/pokemon/cam.png"
    
    print("🚀 YOLO to ONNX Converter with Inference Testing")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Convert the model
    onnx_path = convert_yolo_to_onnx(model_path, output_dir)
    
    if onnx_path:
        print("\n✅ Conversion completed successfully!")
        print(f"📁 ONNX model saved to: {onnx_path}")
        print(f"🌐 Frontend will load from: /models/trading_card_detector.onnx")
        
        # Create a simple info file
        info_path = os.path.join(output_dir, "model_info.json")
        model_info = {
            "name": "Trading Card Detector",
            "version": "1.0.0",
            "format": "ONNX",
            "input_size": 1088,
            "source_model": model_path,
            "classes": ["trading_card"],  # Update with actual class names
            "description": "YOLO11n OBB model for trading card detection"
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📋 Model info saved to: {info_path}")
        
        # Test inference on cam.png
        print(f"\n🧪 Running inference tests...")
        test_results = test_inference_on_image(model_path, onnx_path, test_image_path, output_dir)
        
        if test_results:
            print(f"\n✅ Inference testing completed!")
            print(f"📊 Check {output_dir}/inference_test/ for visualization results")
        else:
            print(f"\n⚠️  Inference testing had issues, but conversion was successful")
        
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
