# YOLO OBB Training Pipeline for Trading Card Detection

This document describes the complete training pipeline for fine-tuning YOLOv11 with Oriented Bounding Boxes (OBB) on the trading card dataset.

## Overview

The pipeline converts COCO-format annotations with rotated bounding boxes to YOLO OBB format and trains a YOLOv11 model optimized for detecting trading cards at arbitrary orientations.

## Dataset Structure

### Input Data (COCO Format)
```
data/training_100k/
├── images/                    # Training images (1024x768)
│   ├── train_000000.jpg
│   ├── train_000001.jpg
│   └── ...
└── annotations/
    ├── train_annotations.json # COCO format with rotated_bbox
    ├── val_annotations.json
    └── test_annotations.json
```

### Output Data (YOLO OBB Format)
```
data/yolo_obb/
├── images/
│   ├── train/ -> ../../training_100k/images
│   ├── val/ -> ../../training_100k/images
│   └── test/ -> ../../training_100k/images
└── labels/
    ├── train/
    │   ├── train_000000.txt
    │   └── ...
    ├── val/
    └── test/
```

## YOLO OBB Label Format

Each label file contains one line per object:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Where:
- `class_id`: 0 (for "card" class)
- `x1 y1 x2 y2 x3 y3 x4 y4`: Normalized coordinates (0-1) of the 4 corners of the oriented bounding box

## Quick Start

### 1. Setup Environment
```bash
# Run the setup script
./scripts/setup_yolo_training.sh
```

This script will:
- Install required packages (ultralytics>=8.2.0)
- Convert COCO annotations to YOLO OBB format
- Download YOLO11 OBB model weights
- Verify dataset structure

### 2. Start Training

#### Quick Test (10 epochs)
```bash
python src/training/train_yolo_obb.py --epochs 10 --batch 8
```

#### Full Training Run
```bash
python src/training/train_yolo_obb.py --epochs 100 --batch 16
```

#### High Accuracy Training
```bash
python src/training/train_yolo_obb.py --model yolo11l-obb.pt --epochs 200 --batch 8
```

## Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolo11n-obb.pt | Nano | Fastest | Good | Quick testing, mobile deployment |
| yolo11s-obb.pt | Small | Fast | Better | Balanced speed/accuracy |
| yolo11m-obb.pt | Medium | Medium | Good | Production deployment |
| yolo11l-obb.pt | Large | Slow | Best | High accuracy requirements |
| yolo11x-obb.pt | Extra Large | Slowest | Highest | Research, maximum accuracy |

## Training Configuration

### Dataset Configuration (`configs/yolo_obb_dataset.yaml`)
- **Image Size**: 1024x768 (matches your dataset)
- **Classes**: 1 (card)
- **Augmentation**: Optimized for OBB with rotation, scaling, HSV adjustments
- **Batch Size**: 16 (adjust based on GPU memory)

### Key Hyperparameters
- **Learning Rate**: 0.01 (initial) → 0.0001 (final)
- **Optimizer**: SGD with momentum 0.937
- **Weight Decay**: 0.0005
- **Warmup**: 3 epochs
- **Early Stopping**: 50 epochs patience

## Training Arguments

```bash
python src/training/train_yolo_obb.py [OPTIONS]

Options:
  --data PATH           Dataset YAML config [default: ../configs/yolo_obb_dataset.yaml]
  --model MODEL         Model architecture [default: yolo11n-obb.pt]
  --epochs INT          Number of epochs [default: 100]
  --imgsz INT           Input image size [default: 640]
  --batch INT           Batch size [default: 16]
  --device DEVICE       Device (0, 1, cpu) [default: auto-detect]
  --project NAME        Project name [default: trading_cards_obb]
  --name NAME           Experiment name [default: yolo11n_obb_v1]
  --resume              Resume from last checkpoint
  --no-pretrained       Train from scratch (no pretrained weights)
  --save-period INT     Save checkpoint every N epochs [default: 10]
  --patience INT        Early stopping patience [default: 50]
  --no-mlflow           Disable MLflow logging
```

## Monitoring Training

### 1. Real-time Logs
Training progress is displayed in the terminal with:
- Loss values (box, cls, dfl, pose)
- mAP metrics (mAP50, mAP50-95)
- Learning rate schedule
- GPU memory usage

### 2. TensorBoard
```bash
tensorboard --logdir outputs/training/
```

### 3. MLflow UI
```bash
mlflow ui --backend-store-uri outputs/mlruns
```

### 4. Training Plots
Automatically saved plots:
- `results.png`: Training/validation metrics over time
- `confusion_matrix.png`: Class confusion matrix
- `F1_curve.png`: F1 score vs confidence threshold
- `PR_curve.png`: Precision-Recall curve

## Output Files

After training, you'll find:
```
outputs/training/[experiment_name]/
├── weights/
│   ├── best.pt          # Best model weights (lowest validation loss)
│   ├── last.pt          # Latest model weights
│   └── epoch_*.pt       # Periodic checkpoints
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png        # F1 score curve
├── PR_curve.png        # Precision-Recall curve
└── args.yaml           # Training arguments used
```

## Performance Expectations

Based on similar OBB detection tasks:

| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) | GPU Memory |
|-------|---------|--------------|------------|------------|
| YOLO11n-OBB | 85-90% | 60-65% | 2-3ms | 2GB |
| YOLO11s-OBB | 88-92% | 65-70% | 3-4ms | 3GB |
| YOLO11m-OBB | 90-94% | 70-75% | 5-6ms | 4GB |
| YOLO11l-OBB | 92-95% | 75-80% | 8-10ms | 6GB |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch 8` or `--batch 4`
   - Use smaller model: `--model yolo11n-obb.pt`
   - Reduce image size: `--imgsz 512`

2. **Slow Training**
   - Increase batch size if GPU memory allows
   - Use multiple GPUs: `--device 0,1`
   - Reduce number of workers in dataset config

3. **Poor Convergence**
   - Check dataset quality and annotations
   - Adjust learning rate: modify `configs/yolo_obb_dataset.yaml`
   - Increase training epochs
   - Try different model architecture

4. **Dataset Conversion Errors**
   - Verify COCO annotation format
   - Check image file paths
   - Ensure rotated_bbox field exists in annotations

### Validation Commands

```bash
# Validate dataset structure
python -c "from ultralytics import YOLO; YOLO().val(data='configs/yolo_obb_dataset.yaml')"

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('yolo11n-obb.pt'); print('Model loaded successfully')"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## Next Steps

1. **Model Evaluation**: Use the trained model for inference on test images
2. **Model Optimization**: Convert to ONNX/TensorRT for deployment
3. **Data Augmentation**: Experiment with additional augmentation techniques
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Active Learning**: Use model predictions to identify hard examples for annotation

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLO OBB Format Specification](https://docs.ultralytics.com/datasets/obb/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
