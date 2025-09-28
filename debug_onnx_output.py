#!/usr/bin/env python3
"""
Debug ONNX output format to understand the exact structure
"""

import cv2
import numpy as np
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

def debug_onnx_output():
    onnx_path = "/home/alec/git/pokemon/frontend/public/models/trading_card_detector.onnx"
    image_path = "/home/alec/git/pokemon/cam.png"
    
    # Load image
    image = cv2.imread(image_path)
    print(f"Original image shape: {image.shape}")
    
    # Preprocess
    letterboxed, ratio, (pad_x, pad_y) = letterbox_resize(image, 1088)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    input_tensor = rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Letterbox ratio: {ratio}, padding: ({pad_x}, {pad_y})")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    
    print(f"\nONNX output shapes: {[output.shape for output in outputs]}")
    
    # Examine the output
    output = outputs[0][0]  # Remove batch dimension
    print(f"Output shape after removing batch: {output.shape}")
    
    # Find non-zero detections
    non_zero_detections = []
    for i in range(output.shape[0]):
        detection = output[i]
        if not np.all(detection == 0):
            non_zero_detections.append((i, detection))
    
    print(f"\nFound {len(non_zero_detections)} non-zero detections:")
    
    for i, (idx, detection) in enumerate(non_zero_detections[:10]):  # Show first 10
        print(f"Detection {i} (index {idx}): {detection}")
        
        if len(detection) == 7:
            print(f"  Parsed as [x1={detection[0]:.4f}, y1={detection[1]:.4f}, x2={detection[2]:.4f}, y2={detection[3]:.4f}, conf={detection[4]:.4f}, class={detection[5]:.4f}, extra={detection[6]:.4f}]")
        elif len(detection) == 6:
            print(f"  Parsed as [x1={detection[0]:.4f}, y1={detection[1]:.4f}, x2={detection[2]:.4f}, y2={detection[3]:.4f}, conf={detection[4]:.4f}, class={detection[5]:.4f}]")
    
    # Check if coordinates are normalized
    if non_zero_detections:
        sample_det = non_zero_detections[0][1]
        print(f"\nSample detection analysis:")
        print(f"  x1={sample_det[0]:.6f}, y1={sample_det[1]:.6f}")
        print(f"  x2={sample_det[2]:.6f}, y2={sample_det[3]:.6f}")
        print(f"  Are coordinates normalized (0-1)? {0 <= sample_det[0] <= 1 and 0 <= sample_det[1] <= 1}")
        print(f"  Are coordinates in pixel space? {sample_det[0] > 1 or sample_det[1] > 1}")

if __name__ == "__main__":
    debug_onnx_output()
