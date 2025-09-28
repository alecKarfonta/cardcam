#!/usr/bin/env python3
"""
Direct comparison between Python ONNX and JavaScript ONNX outputs
to identify exactly where the JavaScript pipeline diverges.
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
import os

def run_python_onnx_inference(model_path, image_path, save_tensors=True):
    """Run ONNX inference in Python and save all intermediate tensors"""
    print("🐍 PYTHON ONNX INFERENCE")
    
    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Original image: {w}x{h}")
    
    # Letterbox preprocessing (exactly matching JavaScript)
    input_size = 1088
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2
    
    print(f"Letterbox: scale={scale:.6f}, new_size={new_w}x{new_h}, pad=({pad_x},{pad_y})")
    
    # Resize and pad
    resized = cv2.resize(img, (new_w, new_h))
    padded = cv2.copyMakeBorder(resized, pad_y, input_size-new_h-pad_y, 
                               pad_x, input_size-new_w-pad_x, 
                               cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    # Convert to tensor (BGR -> RGB for proper comparison)
    rgb_img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb_img.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)   # Add batch
    
    print(f"Input tensor shape: {tensor.shape}")
    print(f"Input tensor range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    
    # Sample some pixels for comparison
    print(f"Sample pixels (center): R={tensor[0,0,544,544]:.6f}, G={tensor[0,1,544,544]:.6f}, B={tensor[0,2,544,544]:.6f}")
    
    if save_tensors:
        # Save input tensor
        np.save('python_input_tensor_debug.npy', tensor)
        print("Saved: python_input_tensor_debug.npy")
    
    # Run ONNX inference
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'images': tensor})
    output = outputs[0]
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Analyze output channels
    if len(output.shape) == 3 and output.shape[1] == 6:
        channels = ['cx', 'cy', 'w', 'h', 'angle', 'conf']
        for i, name in enumerate(channels):
            channel_data = output[0, i, :]
            print(f"Channel {i} ({name}): min={channel_data.min():.3f}, max={channel_data.max():.3f}, mean={channel_data.mean():.3f}")
    
    if save_tensors:
        # Save output tensor
        np.save('python_output_tensor_debug.npy', output)
        print("Saved: python_output_tensor_debug.npy")
    
    # Find high-confidence detections
    if len(output.shape) == 3 and output.shape[1] == 6:
        conf_raw = output[0, 5, :]
        conf_sigmoid = 1 / (1 + np.exp(-conf_raw))
        high_conf_mask = conf_sigmoid > 0.8
        high_conf_indices = np.where(high_conf_mask)[0]
        
        print(f"High confidence detections (>0.8): {len(high_conf_indices)}")
        
        if len(high_conf_indices) > 0:
            # Get top 3 detections
            top_indices = high_conf_indices[np.argsort(conf_sigmoid[high_conf_indices])[-3:]]
            
            detections = []
            for i, idx in enumerate(reversed(top_indices)):
                cx, cy, w, h, angle_raw, conf_raw = output[0, :, idx]
                conf = conf_sigmoid[idx]
                
                # Convert to image coordinates
                img_cx = (cx - pad_x) / scale
                img_cy = (cy - pad_y) / scale
                
                detection = {
                    'rank': i + 1,
                    'anchor_idx': int(idx),
                    'model_coords': {'cx': float(cx), 'cy': float(cy), 'w': float(w), 'h': float(h), 'angle': float(angle_raw)},
                    'confidence': float(conf),
                    'image_coords': {'cx': float(img_cx), 'cy': float(img_cy)},
                    'image_percent': {'cx': float(img_cx/img.shape[1]*100), 'cy': float(img_cy/img.shape[0]*100)}
                }
                detections.append(detection)
                
                print(f"Detection {i+1}: anchor={idx}, conf={conf:.3f}, model=({cx:.1f},{cy:.1f}), image=({img_cx/img.shape[1]*100:.1f}%,{img_cy/img.shape[0]*100:.1f}%)")
        
        # Save detection results
        if save_tensors:
            results = {
                'image_info': {'width': w, 'height': h, 'path': image_path},
                'preprocessing': {'input_size': input_size, 'scale': scale, 'padding': [pad_x, pad_y]},
                'model_output': {'shape': list(output.shape), 'total_anchors': output.shape[2]},
                'detections': detections if 'detections' in locals() else []
            }
            
            with open('python_results_debug.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("Saved: python_results_debug.json")
    
    return output

if __name__ == "__main__":
    model_path = 'frontend/public/models/trading_card_detector_backbone.onnx'
    image_path = 'cam.png'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    try:
        output = run_python_onnx_inference(model_path, image_path)
        print("✅ Python inference completed successfully")
    except Exception as e:
        print(f"❌ Python inference failed: {e}")
        import traceback
        traceback.print_exc()
