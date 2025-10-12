# Training Dataset Manager - User Guide

## Overview

The new dataset management system allows you to build up a collection of training examples, preview them, edit bounding box annotations, and export everything as a batch when ready.

## Quick Start

1. **Open the Camera Test Page**
   ```bash
   # Navigate to frontend/public in your browser
   open http://localhost:8000/camera_test.html
   ```

2. **Capture Training Examples**
   - Click "Start Auto" to begin continuous inference
   - When you see good detections, click "Add to Dataset"
   - The example is saved to your local dataset (browser storage)
   - Continue capturing different scenarios

3. **Manage Your Dataset**
   - Click "Show Dataset Manager" to open the dataset interface
   - View all captured examples in a grid
   - See statistics: total examples, annotations, storage size

4. **Edit Annotations**
   - Click any example thumbnail to open the editor
   - Adjust bounding boxes:
     - **Move**: Click and drag the box
     - **Resize**: Drag corner handles
     - **Rotate**: Drag the circular handle above the box, or use mouse wheel
   - Add new boxes: Click "Add Box" button
   - Delete boxes: Select and click "Delete Selected"
   - Fine-tune: Use the property inputs for precise adjustments

5. **Export Your Dataset**
   - In the Dataset Manager, click "Export Dataset"
   - All images and labels download as separate files
   - Format: YOLO OBB (ready for training)

## Features

### Interactive Editing
- **Visual Handles**: Corner handles for resizing, rotation handle for angle adjustment
- **Mouse Wheel**: Fine rotation control (hold mouse over selected box)
- **Property Editor**: Manual input for exact values
- **Multi-box Support**: Edit multiple annotations per image

### Dataset Management
- **Persistent Storage**: Automatically saved to browser localStorage
- **Statistics Dashboard**: Track your dataset size and quality
- **Bulk Operations**: Export all, clear all, save to storage
- **Preview Thumbnails**: Quick visual reference of all examples

### Quality Control
- **Confidence Filtering**: Only capture high-quality detections
- **Visual Inspection**: Review every example before export
- **Easy Corrections**: Fix any annotation mistakes
- **Delete Bad Examples**: Remove poor quality data

## File Format

### YOLO OBB (Oriented Bounding Box)
Each training example produces two files:

**Image File**: `yolo_000000.jpg`
- Original camera capture
- JPEG format

**Label File**: `yolo_000000.txt`
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
0 0.234 0.156 0.456 0.123 0.489 0.345 0.267 0.378
```
- One line per bounding box
- Normalized coordinates (0-1 range)
- Four corner points (x, y pairs)
- Supports arbitrary rotation

## Tips for High-Quality Training Data

### Variety is Key
- ✅ Multiple angles (straight on, tilted, angled)
- ✅ Different distances (close-up, medium, far)
- ✅ Various lighting (bright, dim, mixed)
- ✅ Different backgrounds (table, fabric, binder)
- ✅ Partial occlusion (cards overlapping)

### Annotation Best Practices
- Tight bounding boxes (minimal background)
- Accurate rotation (aligned with card edges)
- Consistent quality across dataset
- Remove false positives
- Fix low-confidence detections

### Recommended Dataset Size
- Start with: 50-100 examples
- Good baseline: 200-500 examples
- Production quality: 1000+ examples

## Keyboard Shortcuts

Currently not implemented, but planned for future versions:
- `Delete` - Delete selected box
- `Ctrl+Z` - Undo
- `Ctrl+D` - Duplicate box
- `Arrow Keys` - Move box
- `+/-` - Rotate box

## Storage Limits

The dataset is stored in browser localStorage, which has limits:
- Typical limit: ~10MB
- Warning shown when approaching limit
- Recommendation: Export and clear regularly
- Each 1024x768 JPEG ~50-100KB

**Estimated Capacity**: ~100-200 examples depending on image complexity

## Exporting and Backing Up

### Regular Export
1. Click "Export Dataset" in Dataset Manager
2. Save files to organized folder
3. Consider versioning: `dataset_v1`, `dataset_v2`, etc.

### Manual Backup
The dataset is stored in browser localStorage under key: `cv_training_dataset`
- Use browser developer tools to export
- Or use "Save to Storage" button to manually trigger save

### Recommended Workflow
```
1. Collect 50 examples
2. Review and edit
3. Export to folder: dataset_batch_001/
4. Clear dataset or continue
5. Repeat
6. Combine all batches for training
```

## Troubleshooting

### "Storage quota exceeded" error
- Export current dataset
- Click "Clear All" to free space
- Or delete individual examples

### Editor not showing
- Check browser console for errors
- Ensure all script files loaded
- Try refreshing the page

### Bounding boxes look wrong
- Check that model is loaded correctly
- Adjust confidence threshold
- Manually edit in Dataset Manager

### Export not downloading
- Check browser download permissions
- Some browsers block multiple downloads
- Try exporting smaller batches

## Architecture

The system consists of 4 main components:

1. **BoundingBox**: Geometry and format conversion
2. **TrainingExample**: Single image + annotations
3. **DatasetManager**: Collection management and persistence
4. **BoundingBoxEditor**: Interactive canvas editor
5. **DatasetViewer**: UI component

All components follow clean architecture principles with clear separation of concerns.

## Next Steps

After collecting and exporting your dataset:

1. Organize files into YOLO directory structure:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

2. Create YOLO config file (dataset.yaml)

3. Train your model:
   ```bash
   cd training_app
   python src/training/train_yolo_obb.py
   ```

4. Evaluate results and iterate on data collection

## Support

For issues or questions:
- Check `notes.md` for technical details
- Review browser console for errors
- Ensure WebGPU is supported in your browser

## Version History

- v1.0 (2025-10-12): Initial release
  - Core dataset management
  - Interactive bounding box editor
  - YOLO OBB export
  - localStorage persistence

