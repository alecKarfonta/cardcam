# Training Data Generation for Card Segmentation

This document describes the synthetic training data generation system for creating segmentation training examples from individual card images.

## Overview

The training data generation system creates realistic multi-card scenes from our collection of individual card images. This allows us to generate thousands of training examples with perfect ground truth annotations for segmentation model training.

## Features

### 🎨 **Scene Generation**
- **Multi-card layouts**: 1-8 cards per scene with configurable parameters
- **Realistic positioning**: Smart placement with optional overlap detection
- **Various backgrounds**: Solid colors, gradients, textures, wood, and fabric patterns
- **Flexible output sizes**: Configurable scene dimensions (default: 1024x768)

### 🔄 **Card Transformations**
- **Rotation**: Random rotation within specified range (±15° default)
- **Scaling**: Adaptive scaling based on card size constraints
- **Perspective**: 3D perspective transformations for realism
- **Augmentations**: Brightness, contrast, hue, saturation, noise, and blur

### 📊 **Annotation Generation**
- **COCO Format**: Industry-standard annotations for segmentation
- **Precise masks**: Pixel-perfect segmentation masks
- **Bounding boxes**: Accurate bounding box coordinates
- **Instance separation**: Individual annotations for each card

### 🎯 **Quality Control**
- **Overlap management**: Configurable overlap probability and limits
- **Size constraints**: Minimum and maximum card sizes
- **Area validation**: Ensures sufficient card visibility
- **Mask quality**: Validates segmentation mask integrity

## Usage

### Basic Usage

```bash
# Generate training data with default settings
python scripts/generate_training_data.py

# Generate with custom parameters
python scripts/generate_training_data.py \
    --card_images_dir data/raw/card_images \
    --output_dir data/training_scenes \
    --num_train 2000 \
    --num_val 400 \
    --min_cards 2 \
    --max_cards 6
```

### Advanced Configuration

```bash
# Custom scene parameters
python scripts/generate_training_data.py \
    --output_size 1280,960 \
    --min_cards 1 \
    --max_cards 8 \
    --card_images_dir data/raw/card_images \
    --output_dir data/custom_training
```

### Visualization

```bash
# Visualize generated training data
python scripts/visualize_training_data.py \
    --data_dir data/training_scenes \
    --num_samples 10 \
    --show_stats

# Visualize validation data
python scripts/visualize_training_data.py \
    --data_dir data/training_scenes \
    --split val \
    --num_samples 5
```

## Configuration

### Scene Configuration (`SceneConfig`)

```python
@dataclass
class SceneConfig:
    output_size: Tuple[int, int] = (1024, 768)  # Width x Height
    min_cards: int = 1                          # Minimum cards per scene
    max_cards: int = 8                          # Maximum cards per scene
    min_card_size: int = 150                    # Minimum card dimension
    max_card_size: int = 300                    # Maximum card dimension
    overlap_probability: float = 0.3            # Probability of overlap
    max_overlap: float = 0.4                    # Maximum overlap ratio
    rotation_range: Tuple[float, float] = (-15, 15)  # Rotation range
    perspective_probability: float = 0.4        # Perspective transform prob
    lighting_probability: float = 0.6           # Lighting effects prob
```

### YAML Configuration

See `configs/training_generation.yaml` for detailed configuration options:

```yaml
scene_generation:
  output_size: [1024, 768]
  min_cards: 1
  max_cards: 6
  overlap_probability: 0.3
  max_overlap: 0.4

augmentations:
  brightness_contrast:
    brightness_limit: 0.2
    contrast_limit: 0.2
    probability: 0.7
```

## Output Format

### Directory Structure

```
output_dir/
├── images/
│   ├── train_000000.jpg
│   ├── train_000001.jpg
│   └── ...
└── annotations/
    ├── train_annotations.json
    └── val_annotations.json
```

### COCO Annotation Format

```json
{
  "info": {
    "description": "Trading Card Segmentation Dataset",
    "version": "1.0",
    "year": 2025
  },
  "images": [
    {
      "id": 1,
      "width": 1024,
      "height": 768,
      "file_name": "train_000000.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 32276,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "card",
      "supercategory": "object"
    }
  ]
}
```

## Background Generation

The system generates diverse backgrounds to improve model robustness:

### Background Types

1. **Solid Colors**: Light, neutral colors suitable for card photography
2. **Gradients**: Smooth color transitions
3. **Textures**: Noise-based textured surfaces
4. **Wood Patterns**: Wood grain-like backgrounds
5. **Fabric Textures**: Fabric-like surface patterns

### Custom Backgrounds

You can provide custom background images by placing them in a directory and specifying the path:

```python
background_generator = BackgroundGenerator("path/to/backgrounds")
```

## Augmentation Pipeline

### Card-Level Augmentations

- **Brightness/Contrast**: ±20% adjustment with 70% probability
- **Color Jitter**: Hue (±10°), Saturation (±20%), Value (±20%)
- **Noise**: Gaussian noise with configurable variance
- **Blur**: Gaussian blur up to 3px radius

### Geometric Transformations

- **Rotation**: Random rotation within specified range
- **Scaling**: Maintains card aspect ratio while fitting size constraints
- **Perspective**: 3D perspective transformation for depth realism

### Scene-Level Effects

- **Lighting**: Global brightness and contrast adjustments
- **Atmospheric**: Slight blur for camera focus effects

## Quality Metrics

### Generated Dataset Statistics

The system tracks and reports:

- **Cards per image distribution**
- **Card area distribution**
- **Bounding box size distribution**
- **Overlap statistics**
- **Annotation quality metrics**

### Validation Checks

- **Mask integrity**: Ensures segmentation masks are valid
- **Bounding box accuracy**: Verifies bbox coordinates
- **Area consistency**: Checks mask area vs. calculated area
- **Visibility requirements**: Ensures minimum card visibility

## Performance Considerations

### Memory Usage

- **Image processing**: ~100MB per 1024x768 scene
- **Batch generation**: Configurable batch sizes
- **Memory limits**: Automatic cleanup and garbage collection

### Generation Speed

- **Typical performance**: 2-5 images per second
- **Factors affecting speed**:
  - Image resolution
  - Number of cards per scene
  - Augmentation complexity
  - Disk I/O speed

### Optimization Tips

1. **Use SSD storage** for faster I/O
2. **Adjust batch sizes** based on available RAM
3. **Reduce augmentation complexity** for faster generation
4. **Use multiple workers** for parallel processing

## Integration with Training Pipeline

### Detectron2 Integration

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register dataset
register_coco_instances(
    "card_train", 
    {}, 
    "data/training_scenes/annotations/train_annotations.json",
    "data/training_scenes/images"
)

# Use in training
cfg.DATASETS.TRAIN = ("card_train",)
```

### YOLOv11 Integration

```python
# Convert COCO to YOLO format
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="data/training_scenes/annotations/",
    save_dir="data/yolo_format/"
)
```

## Troubleshooting

### Common Issues

1. **Low card visibility**: Increase `min_card_size` or reduce `max_cards`
2. **Poor mask quality**: Check input card image quality
3. **Memory errors**: Reduce batch size or image resolution
4. **Slow generation**: Use faster storage or reduce augmentations

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Quality Inspection

Use the visualization script to inspect generated data:

```bash
python scripts/visualize_training_data.py --show_stats
```

## Future Enhancements

### Planned Features

- **Real background integration**: Use actual table/surface photos
- **Advanced lighting models**: Realistic shadow and reflection effects
- **Card condition simulation**: Wear, damage, and aging effects
- **Multi-game mixing**: Scenes with cards from different games
- **Binder/sleeve simulation**: Cards in protective sleeves or binders

### Performance Improvements

- **GPU acceleration**: CUDA-based image processing
- **Parallel generation**: Multi-process scene generation
- **Streaming generation**: On-demand data generation during training
- **Caching system**: Reuse transformed cards across scenes

## Examples

### Basic Generation

```python
from scripts.generate_training_data import TrainingDataGenerator, SceneConfig

config = SceneConfig(
    output_size=(1024, 768),
    min_cards=2,
    max_cards=5
)

generator = TrainingDataGenerator(
    card_images_dir="data/raw/card_images",
    output_dir="data/training",
    config=config
)

# Generate 1000 training images
results = generator.generate_dataset(1000, "train")
```

### Custom Scene Generation

```python
from scripts.generate_training_data import SceneGenerator, BackgroundGenerator

# Custom background generator
bg_gen = BackgroundGenerator("custom_backgrounds/")

# Custom scene generator
scene_gen = SceneGenerator("data/raw/card_images", config, bg_gen)

# Generate single scene
scene, instances = scene_gen.generate_scene()
```

This system provides a robust foundation for generating high-quality training data for card segmentation models, with extensive customization options and quality control measures.
