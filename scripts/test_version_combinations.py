#!/usr/bin/env python3
"""
Test different PyTorch/Ultralytics version combinations for ONNX export
"""

import subprocess
import sys
import os
from pathlib import Path

# Known working combinations based on research
VERSION_COMBINATIONS = [
    {
        "name": "Stable PyTorch 2.3.1 + Ultralytics 8.2.x",
        "torch": "2.3.1",
        "torchvision": "0.18.1", 
        "ultralytics": "8.2.103",
        "description": "Known stable combination for ONNX export"
    },
    {
        "name": "PyTorch 2.4.1 + Latest Ultralytics",
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "ultralytics": "8.3.67",
        "description": "More recent stable versions"
    },
    {
        "name": "PyTorch 2.2.2 + Ultralytics 8.1.x", 
        "torch": "2.2.2",
        "torchvision": "0.17.2",
        "ultralytics": "8.1.47",
        "description": "Older stable combination"
    }
]

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr

def install_version_combination(combo):
    """Install a specific version combination"""
    print(f"\n{'='*60}")
    print(f"Installing: {combo['name']}")
    print(f"Description: {combo['description']}")
    print(f"{'='*60}")
    
    # Uninstall existing packages
    print("\n1. Uninstalling existing packages...")
    uninstall_cmd = "pip uninstall -y torch torchvision ultralytics"
    success, output = run_command(uninstall_cmd, "Uninstalling existing packages")
    
    # Install specific versions
    print("\n2. Installing specific versions...")
    
    # Install PyTorch and torchvision
    torch_cmd = f"pip install torch=={combo['torch']} torchvision=={combo['torchvision']} --index-url https://download.pytorch.org/whl/cu118"
    success, output = run_command(torch_cmd, f"Installing PyTorch {combo['torch']} and torchvision {combo['torchvision']}")
    if not success:
        print("❌ Failed to install PyTorch/torchvision")
        return False
    
    # Install ultralytics
    ultralytics_cmd = f"pip install ultralytics=={combo['ultralytics']}"
    success, output = run_command(ultralytics_cmd, f"Installing Ultralytics {combo['ultralytics']}")
    if not success:
        print("❌ Failed to install Ultralytics")
        return False
    
    # Verify installation
    print("\n3. Verifying installation...")
    verify_cmd = "python -c \"import torch; import ultralytics; print(f'PyTorch: {torch.__version__}'); print(f'Ultralytics: {ultralytics.__version__}')\""
    success, output = run_command(verify_cmd, "Verifying installation")
    if success:
        print(f"Installation output:\n{output}")
    
    return success

def test_onnx_export(model_path, output_dir):
    """Test ONNX export with current versions"""
    print(f"\n{'='*40}")
    print("Testing ONNX Export")
    print(f"{'='*40}")
    
    test_script = f"""
import os
from ultralytics import YOLO

try:
    print("Loading model...")
    model = YOLO('{model_path}')
    
    print("Attempting ONNX export...")
    success = model.export(
        format="onnx",
        imgsz=1088,
        optimize=True,
        half=False,
        dynamic=False,
        simplify=True,
        nms=True,
        opset=17,  # Try older opset
    )
    
    if success:
        print("✅ ONNX export successful!")
        exported_path = '{model_path}'.replace('.pt', '.onnx')
        if os.path.exists(exported_path):
            print(f"✅ ONNX file created: {{exported_path}}")
            file_size = os.path.getsize(exported_path) / (1024 * 1024)
            print(f"📊 File size: {{file_size:.2f}} MB")
        else:
            print("⚠️ Export reported success but file not found")
    else:
        print("❌ Export failed")
        
except Exception as e:
    print(f"❌ Export failed with error: {{e}}")
    import traceback
    traceback.print_exc()
"""
    
    # Write test script to file
    test_file = "/tmp/test_export.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    # Run test
    success, output = run_command(f"python {test_file}", "Testing ONNX export")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return success, output

def main():
    print("🧪 PyTorch/Ultralytics Version Compatibility Tester")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Please activate your virtual environment first.")
            return
    
    model_path = "/home/alec/git/pokemon/src/training/trading_cards_obb/yolo11n_obb_v16/weights/best.pt"
    output_dir = "/home/alec/git/pokemon/frontend/public/models"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Test each version combination
    successful_combinations = []
    
    for i, combo in enumerate(VERSION_COMBINATIONS, 1):
        print(f"\n🧪 Testing combination {i}/{len(VERSION_COMBINATIONS)}")
        
        # Install version combination
        if install_version_combination(combo):
            # Test ONNX export
            success, output = test_onnx_export(model_path, output_dir)
            
            if success:
                print(f"✅ SUCCESS: {combo['name']} works for ONNX export!")
                successful_combinations.append(combo)
                
                # Ask if user wants to keep this version
                response = input(f"\n🎯 This combination works! Keep it? (Y/n): ")
                if response.lower() != 'n':
                    print(f"✅ Keeping {combo['name']}")
                    break
            else:
                print(f"❌ FAILED: {combo['name']} does not work for ONNX export")
        else:
            print(f"❌ FAILED: Could not install {combo['name']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if successful_combinations:
        print("✅ Working combinations found:")
        for combo in successful_combinations:
            print(f"  - {combo['name']}")
            print(f"    PyTorch: {combo['torch']}, Ultralytics: {combo['ultralytics']}")
    else:
        print("❌ No working combinations found")
        print("Consider trying alternative export methods or older versions")
    
    print("\n🔄 You can run this script again to test more combinations")

if __name__ == "__main__":
    main()
