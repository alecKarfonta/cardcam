# Model Registry

This directory contains the model registry (`models.json`) that defines all available models for inference in the WebGPU Camera Test application.

## Model Metadata Format

Each model in the registry must include the following fields:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the model (use snake_case) |
| `name` | string | Human-readable display name |
| `path` | string | Path to the ONNX model file relative to the web root |
| `type` | string | Model type (e.g., `obb`, `bbox`, `segmentation`, `classification`) |
| `resolution` | number | Input resolution in pixels (e.g., `640`, `1088`) |
| `architecture` | string | Model architecture (e.g., `YOLOv12n`, `YOLOv8m`, `ResNet50`) |
| `classes` | array[string] | List of class names the model can detect |
| `classCount` | number | Number of classes (should match length of `classes` array) |
| `description` | string | Detailed description of the model and its use case |
| `outputFormat` | string | Output format identifier (e.g., `obb`, `bbox`, `mask`) |
| `valuesPerDetection` | number | Number of values per detection in model output |
| `coordinateFormat` | string | Coordinate format description (e.g., `center_xywh_angle`, `xyxy`) |

### Example Model Entry

```json
{
  "id": "trading_card_obb_v1",
  "name": "Trading Card Detector (OBB)",
  "path": "/models/trading_card_detector.onnx",
  "type": "obb",
  "resolution": 1088,
  "architecture": "YOLOv12n",
  "classes": ["trading_card"],
  "classCount": 1,
  "description": "Single-class oriented bounding box model for detecting trading cards at any angle. Trained on YOLOv12n architecture with 1088px input resolution.",
  "outputFormat": "obb",
  "valuesPerDetection": 7,
  "coordinateFormat": "center_xywh_angle"
}
```

## Model Types

### Supported Model Types

- **`obb`** - Oriented Bounding Box detection (includes rotation angle)
- **`bbox`** - Standard axis-aligned bounding box detection
- **`segmentation`** - Instance or semantic segmentation
- **`classification`** - Image classification
- **`pose`** - Pose estimation (keypoint detection)

### Output Formats

#### OBB (Oriented Bounding Box)
- **valuesPerDetection**: 7
- **coordinateFormat**: `center_xywh_angle`
- **Output structure**: `[x_center, y_center, width, height, confidence, class_id, angle]`

#### BBOX (Standard Bounding Box)
- **valuesPerDetection**: 6
- **coordinateFormat**: `center_xywh` or `xyxy`
- **Output structure**: `[x, y, w, h, confidence, class_id]` or `[x1, y1, x2, y2, confidence, class_id]`

## Adding a New Model

1. Place your ONNX model file in the `/models/` directory
2. Add a new entry to the `models` array in `models.json`
3. Ensure all required fields are populated
4. Test the model by selecting it from the Model Selection panel in the UI

## Filtering and Search

The Model Selection panel supports filtering by:
- **Search**: Searches across model name, description, architecture, and class names
- **Type**: Filter by model type (OBB, BBOX, etc.)
- **Architecture**: Filter by neural network architecture
- **Resolution**: Filter by input resolution

## Notes

- Models are automatically discovered from `models.json` on application startup
- The first model in the array is loaded by default
- All models must be WebGPU-compatible ONNX models
- Coordinate formats should match the preprocessing and postprocessing logic in the application

