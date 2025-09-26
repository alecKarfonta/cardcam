#!/bin/bash

# Trading Card Segmentation Environment Setup Script
# This script sets up the development environment without Docker for local development

set -e  # Exit on any error

echo "🚀 Setting up Trading Card Segmentation development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible (>= $required_version)"
else
    echo "❌ Python $python_version is not compatible. Please install Python >= $required_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install basic requirements first
echo "📚 Installing basic Python packages..."
pip install wheel setuptools

# Install PyTorch with CUDA support (if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other ML packages
echo "🧠 Installing ML and computer vision packages..."
pip install opencv-python-headless scikit-image Pillow numpy scipy
pip install albumentations imgaug pandas scikit-learn
pip install matplotlib seaborn plotly tqdm

# Install web framework and API packages
echo "🌐 Installing web framework and API packages..."
pip install fastapi uvicorn requests aiohttp pydantic

# Install development tools
echo "🛠️ Installing development tools..."
pip install jupyter jupyterlab ipywidgets
pip install pytest pytest-cov black flake8 mypy
pip install python-dotenv click rich typer pyyaml toml

# Install ML experiment tracking
echo "📊 Installing experiment tracking tools..."
pip install mlflow wandb tensorboard

# Try to install Detectron2 (may fail without proper CUDA setup)
echo "🔍 Attempting to install Detectron2..."
if pip install 'git+https://github.com/facebookresearch/detectron2.git'; then
    echo "✅ Detectron2 installed successfully"
else
    echo "⚠️ Detectron2 installation failed - will need manual setup later"
fi

# Install YOLO
echo "🎯 Installing YOLO..."
pip install ultralytics

# Install API clients
echo "🃏 Installing card game API clients..."
pip install pokemontcgsdk mtgsdk

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,annotations}/card_images
mkdir -p logs models checkpoints outputs
mkdir -p data/raw/{mtg,pokemon,yugioh}

# Set up environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp env.example .env
    echo "📝 Created .env file from template - please edit with your API keys"
fi

# Initialize Jupyter configuration
echo "📓 Setting up Jupyter..."
jupyter notebook --generate-config --allow-root 2>/dev/null || true

# Test basic imports
echo "🧪 Testing basic imports..."
python3 -c "
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ Basic imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Start data collection: python src/data/api_clients.py"
echo "4. Launch Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo ""
echo "If you have Docker available, you can also use: docker-compose up -d"
