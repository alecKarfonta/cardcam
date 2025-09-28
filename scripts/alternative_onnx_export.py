#!/usr/bin/env python3
"""
Alternative ONNX export methods for YOLO models
Tries different approaches when standard export fails
"""

import os
import sys
import torch
import traceback
from pathlib import Path
from ultralytics import YOLO

def try_export_method_1(model, output_path):
    """Standard export with different opset versions"""
    print("🔄 Method 1: Standard export with different opset versions")
    
    opsets = [17, 16, 15, 14, 13, 12, 11]
    
    for opset in opsets:
        try:
            print(f"  Trying opset {opset}...")
            success = model.export(
                format="onnx",
                imgsz=1088,
                optimize=True,
                half=False,
                dynamic=False,
                simplify=True,
                nms=False,  # Disable NMS first
                opset=opset,
            )
            
            if success:
                print(f"✅ Success with opset {opset}")
                return True, opset
                
        except Exception as e:
            print(f"  ❌ Opset {opset} failed: {str(e)[:100]}...")
            continue
    
    return False, None

def try_export_method_2(model, output_path):
    """Export with minimal settings"""
    print("🔄 Method 2: Minimal export settings")
    
    try:
        success = model.export(
            format="onnx",
            imgsz=640,  # Smaller size
            optimize=False,  # No optimization
            half=False,
            dynamic=False,
            simplify=False,  # No simplification
            nms=False,
            opset=11,  # Very old opset
        )
        
        if success:
            print("✅ Success with minimal settings")
            return True, "minimal"
            
    except Exception as e:
        print(f"❌ Minimal export failed: {e}")
        return False, None

def try_export_method_3(model, output_path):
    """Export with torch.onnx directly (bypass ultralytics)"""
    print("🔄 Method 3: Direct torch.onnx export")
    
    try:
        # Get the model in eval mode
        model.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 1088, 1088)
        
        # Export using torch.onnx directly
        torch.onnx.export(
            model.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        if os.path.exists(output_path):
            print("✅ Success with direct torch.onnx export")
            return True, "direct_torch"
        else:
            print("❌ Direct export failed - no output file")
            return False, None
            
    except Exception as e:
        print(f"❌ Direct torch.onnx export failed: {e}")
        return False, None

def try_export_method_4(model, output_path):
    """Export with TorchScript intermediate step"""
    print("🔄 Method 4: TorchScript -> ONNX export")
    
    try:
        # First export to TorchScript
        print("  Step 1: Exporting to TorchScript...")
        ts_path = output_path.replace('.onnx', '.torchscript')
        
        success = model.export(
            format="torchscript",
            imgsz=1088,
            optimize=False,
            half=False,
        )
        
        if success and os.path.exists(ts_path):
            print("  Step 2: Converting TorchScript to ONNX...")
            
            # Load TorchScript model
            ts_model = torch.jit.load(ts_path)
            ts_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 1088, 1088)
            
            # Export TorchScript to ONNX
            torch.onnx.export(
                ts_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            # Clean up TorchScript file
            if os.path.exists(ts_path):
                os.remove(ts_path)
            
            if os.path.exists(output_path):
                print("✅ Success with TorchScript intermediate")
                return True, "torchscript"
            else:
                print("❌ TorchScript->ONNX failed - no output file")
                return False, None
        else:
            print("❌ TorchScript export failed")
            return False, None
            
    except Exception as e:
        print(f"❌ TorchScript->ONNX export failed: {e}")
        return False, None

def try_export_method_5(model, output_path):
    """Export with older PyTorch compatibility mode"""
    print("🔄 Method 5: Compatibility mode export")
    
    try:
        # Set environment variables for compatibility
        os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
        
        success = model.export(
            format="onnx",
            imgsz=1088,
            optimize=False,
            half=False,
            dynamic=False,
            simplify=False,
            nms=False,
            opset=11,
            verbose=True,
        )
        
        if success:
            print("✅ Success with compatibility mode")
            return True, "compatibility"
            
    except Exception as e:
        print(f"❌ Compatibility mode export failed: {e}")
        return False, None
    finally:
        # Clean up environment variables
        os.environ.pop('PYTORCH_JIT_USE_NNC_NOT_NVFUSER', None)
        os.environ.pop('TORCH_SHOW_CPP_STACKTRACES', None)

def alternative_onnx_export(model_path, output_dir):
    """Try multiple export methods until one succeeds"""
    print("🚀 Alternative ONNX Export Methods")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "trading_card_detector_alt.onnx")
    
    # Try different export methods
    methods = [
        try_export_method_1,
        try_export_method_2,
        try_export_method_3,
        try_export_method_4,
        try_export_method_5,
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n{'='*40}")
        print(f"Trying method {i}/{len(methods)}")
        print(f"{'='*40}")
        
        try:
            success, method_info = method(model, output_path)
            
            if success:
                # Check if file was actually created
                exported_path = str(model_path).replace('.pt', '.onnx')
                if os.path.exists(exported_path):
                    # Move to our desired location
                    import shutil
                    shutil.move(exported_path, output_path)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"\n🎉 SUCCESS!")
                    print(f"📁 ONNX file: {output_path}")
                    print(f"📊 File size: {file_size:.2f} MB")
                    print(f"🔧 Method: {method_info}")
                    
                    # Test loading the ONNX file
                    try:
                        import onnxruntime as ort
                        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
                        inputs = session.get_inputs()
                        outputs = session.get_outputs()
                        
                        print(f"✅ ONNX file validation successful")
                        print(f"   Inputs: {[inp.name + str(inp.shape) for inp in inputs]}")
                        print(f"   Outputs: {[out.name + str(out.shape) for out in outputs]}")
                        
                        return output_path
                        
                    except Exception as e:
                        print(f"⚠️  ONNX file created but validation failed: {e}")
                        return output_path  # Still return path, might work
                else:
                    print(f"❌ Method reported success but no file found")
            else:
                print(f"❌ Method {i} failed")
                
        except Exception as e:
            print(f"❌ Method {i} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n💥 All export methods failed!")
    print(f"Consider trying different PyTorch/Ultralytics versions")
    return None

def main():
    model_path = "/home/alec/git/pokemon/src/training/trading_cards_obb/yolo11n_obb_v16/weights/best.pt"
    output_dir = "/home/alec/git/pokemon/frontend/public/models"
    
    print("Current PyTorch version:", torch.__version__)
    
    result = alternative_onnx_export(model_path, output_dir)
    
    if result:
        print(f"\n✅ Export successful: {result}")
        sys.exit(0)
    else:
        print(f"\n❌ All export methods failed")
        print(f"Next steps:")
        print(f"1. Try running scripts/test_version_combinations.py")
        print(f"2. Downgrade to PyTorch 2.3.1 + Ultralytics 8.2.103")
        sys.exit(1)

if __name__ == "__main__":
    main()
