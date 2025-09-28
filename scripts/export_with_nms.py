#!/usr/bin/env python3
"""
Try to export YOLO model with NMS enabled using different approaches
"""

import os
import sys
import torch
from ultralytics import YOLO

def try_nms_export_method_1(model, output_path):
    """Try NMS export with different opset versions"""
    print("🔄 Method 1: NMS export with different opset versions")
    
    # Try opsets that support NMS operations
    opsets = [16, 15, 14, 13, 12, 11]
    
    for opset in opsets:
        try:
            print(f"  Trying opset {opset} with NMS...")
            success = model.export(
                format="onnx",
                imgsz=1088,
                optimize=True,
                half=False,
                dynamic=False,
                simplify=True,
                nms=True,  # Enable NMS
                opset=opset,
                verbose=True
            )
            
            if success:
                print(f"✅ Success with opset {opset} + NMS")
                return True, opset
                
        except Exception as e:
            print(f"  ❌ Opset {opset} failed: {str(e)[:150]}...")
            continue
    
    return False, None

def try_nms_export_method_2(model, output_path):
    """Try NMS export with minimal optimization"""
    print("🔄 Method 2: NMS export with minimal optimization")
    
    try:
        success = model.export(
            format="onnx",
            imgsz=640,  # Smaller input size
            optimize=False,  # No optimization
            half=False,
            dynamic=False,
            simplify=False,  # No simplification
            nms=True,  # Enable NMS
            opset=11,  # Very conservative opset
            verbose=True
        )
        
        if success:
            print("✅ Success with minimal settings + NMS")
            return True, "minimal_nms"
            
    except Exception as e:
        print(f"❌ Minimal NMS export failed: {e}")
        return False, None

def try_nms_export_method_3(model, output_path):
    """Try export with specific NMS parameters"""
    print("🔄 Method 3: NMS export with custom parameters")
    
    try:
        # Set specific NMS parameters
        success = model.export(
            format="onnx",
            imgsz=1088,
            optimize=True,
            half=False,
            dynamic=False,
            simplify=False,  # Don't simplify when using NMS
            nms=True,
            opset=16,  # Try 16 specifically
            conf=0.25,  # Set confidence threshold
            iou=0.45,   # Set IoU threshold for NMS
            verbose=True
        )
        
        if success:
            print("✅ Success with custom NMS parameters")
            return True, "custom_nms"
            
    except Exception as e:
        print(f"❌ Custom NMS export failed: {e}")
        return False, None

def try_nms_export_method_4(model, output_path):
    """Try export without simplification but with NMS"""
    print("🔄 Method 4: NMS export without graph simplification")
    
    opsets = [17, 16, 15, 14]
    
    for opset in opsets:
        try:
            print(f"  Trying opset {opset} with NMS (no simplify)...")
            success = model.export(
                format="onnx",
                imgsz=1088,
                optimize=False,  # No optimization
                half=False,
                dynamic=False,
                simplify=False,  # Critical: no simplification
                nms=True,
                opset=opset,
                verbose=True
            )
            
            if success:
                print(f"✅ Success with opset {opset} + NMS (no simplify)")
                return True, f"opset{opset}_no_simplify"
                
        except Exception as e:
            print(f"  ❌ Opset {opset} failed: {str(e)[:100]}...")
            continue
    
    return False, None

def main():
    model_path = "/home/alec/git/pokemon/src/training/trading_cards_obb/yolo11n_obb_v16/weights/best.pt"
    output_dir = "/home/alec/git/pokemon/frontend/public/models"
    
    print("🚀 YOLO ONNX Export with NMS")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "trading_card_detector_nms.onnx")
    
    # Try different NMS export methods
    methods = [
        try_nms_export_method_1,
        try_nms_export_method_2, 
        try_nms_export_method_3,
        try_nms_export_method_4,
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'='*40}")
        print(f"Trying NMS method {i}/{len(methods)}")
        print(f"{'='*40}")
        
        try:
            success, method_info = method(model, output_path)
            
            if success:
                # Check if file was created
                exported_path = str(model_path).replace('.pt', '.onnx')
                if os.path.exists(exported_path):
                    import shutil
                    shutil.move(exported_path, output_path)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"\n🎉 NMS EXPORT SUCCESS!")
                    print(f"📁 ONNX file: {output_path}")
                    print(f"📊 File size: {file_size:.2f} MB")
                    print(f"🔧 Method: {method_info}")
                    
                    # Test the ONNX file
                    try:
                        import onnxruntime as ort
                        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
                        inputs = session.get_inputs()
                        outputs = session.get_outputs()
                        
                        print(f"✅ ONNX validation successful")
                        print(f"   Inputs: {[inp.name + str(inp.shape) for inp in inputs]}")
                        print(f"   Outputs: {[out.name + str(out.shape) for out in outputs]}")
                        
                        # Check if this looks like NMS output (should have fewer detections)
                        if len(outputs) > 1:
                            print(f"🎯 Multiple outputs detected - likely includes NMS!")
                        
                        return output_path
                        
                    except Exception as e:
                        print(f"⚠️  ONNX validation failed: {e}")
                        return output_path
                else:
                    print(f"❌ Method reported success but no file found")
            
        except Exception as e:
            print(f"❌ Method {i} crashed: {e}")
    
    print(f"\n💥 All NMS export methods failed!")
    print(f"Will need to implement NMS in frontend JavaScript")
    return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ NMS-enabled ONNX export successful!")
        sys.exit(0)
    else:
        print(f"\n❌ NMS export failed - need frontend NMS implementation")
        sys.exit(1)
