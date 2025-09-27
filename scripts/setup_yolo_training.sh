#!/bin/bash
# Setup script for YOLO OBB training pipeline

set -e  # Exit on any error

echo "🚀 Setting up YOLO OBB Training Pipeline for Trading Cards"
echo "============================================================"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider activating one:"
    echo "   source venv/bin/activate"
    echo ""
fi

# Install/upgrade required packages
echo "📦 Installing required packages..."
pip install --upgrade ultralytics>=8.2.0
pip install --upgrade mlflow
pip install --upgrade pyyaml
pip install --upgrade tqdm

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# Create necessary directories
echo "📁 Creating output directories..."
mkdir -p "$PROJECT_ROOT/outputs/training"
mkdir -p "$PROJECT_ROOT/outputs/mlruns"
mkdir -p "$PROJECT_ROOT/data/yolo_obb"

# Check if training data exists
TRAINING_DATA="$PROJECT_ROOT/data/training_100k"
if [[ ! -d "$TRAINING_DATA" ]]; then
    echo "❌ Error: Training data not found at $TRAINING_DATA"
    echo "   Please ensure your training_100k dataset is available."
    exit 1
fi

echo "✅ Training data found: $TRAINING_DATA"

# Check annotation files
for split in train val test; do
    annotation_file="$TRAINING_DATA/annotations/${split}_annotations.json"
    if [[ -f "$annotation_file" ]]; then
        echo "✅ Found $split annotations"
    else
        echo "⚠️  Warning: Missing $split annotations at $annotation_file"
    fi
done

# Convert COCO to YOLO OBB format
echo "🔄 Converting COCO annotations to YOLO OBB format..."
cd "$PROJECT_ROOT"
python src/training/convert_coco_to_yolo_obb.py

# Verify YOLO dataset structure
YOLO_DATA="$PROJECT_ROOT/data/yolo_obb"
if [[ -d "$YOLO_DATA" ]]; then
    echo "✅ YOLO OBB dataset created at: $YOLO_DATA"
    
    # Count files in each split
    for split in train val test; do
        if [[ -d "$YOLO_DATA/labels/$split" ]]; then
            label_count=$(find "$YOLO_DATA/labels/$split" -name "*.txt" | wc -l)
            echo "   📊 $split: $label_count label files"
        fi
    done
else
    echo "❌ Error: Failed to create YOLO OBB dataset"
    exit 1
fi

# Test model download
echo "🤖 Testing YOLO model download..."
python -c "from ultralytics import YOLO; model = YOLO('yolo11n-obb.pt'); print('✅ YOLO11n-OBB model downloaded successfully')"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Review the dataset configuration: configs/yolo_obb_dataset.yaml"
echo "2. Start training with:"
echo "   cd $PROJECT_ROOT"
echo "   python src/training/train_yolo_obb.py"
echo ""
echo "🔧 Training options:"
echo "   # Quick test run (10 epochs)"
echo "   python src/training/train_yolo_obb.py --epochs 10 --batch 8"
echo ""
echo "   # Full training run"
echo "   python src/training/train_yolo_obb.py --epochs 100 --batch 16"
echo ""
echo "   # Large model for best accuracy"
echo "   python src/training/train_yolo_obb.py --model yolo11l-obb.pt --epochs 200"
echo ""
echo "📊 Monitor training:"
echo "   - Tensorboard logs: outputs/training/[experiment_name]/weights/"
echo "   - MLflow UI: mlflow ui --backend-store-uri outputs/mlruns"
echo ""
echo "🎯 Dataset info:"
echo "   - Format: YOLO OBB (Oriented Bounding Boxes)"
echo "   - Classes: 1 (card)"
echo "   - Image size: 1024x768"
echo "   - Location: $YOLO_DATA"
