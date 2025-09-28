#!/bin/bash
# Setup script for Pokemon Trading Card Detection project

echo "🚀 Setting up Pokemon Trading Card Detection Environment"
echo "=" * 60

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run training with enhanced augmentations:"
echo "  cd src/training"
echo "  python train_yolo_obb.py --data ../../configs/yolo_obb_dataset.yaml --aug-intensity medium"
echo ""
echo "To run training without enhanced augmentations:"
echo "  python train_yolo_obb.py --data ../../configs/yolo_obb_dataset.yaml --no-albumentations"
