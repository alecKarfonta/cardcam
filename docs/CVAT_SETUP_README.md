# CVAT Local Deployment for Trading Card Annotation

## Overview

This document provides instructions for using the locally deployed CVAT (Computer Vision Annotation Tool) instance for annotating trading card images. CVAT is now configured and ready for creating high-quality training data for the card segmentation project.

## Quick Start

### 1. Access CVAT
- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin123`

### 2. Management Commands

Use the provided management script for easy CVAT operations:

```bash
# Check CVAT status
./scripts/manage_cvat.sh status

# Start CVAT (if stopped)
./scripts/manage_cvat.sh start

# Stop CVAT
./scripts/manage_cvat.sh stop

# Restart CVAT
./scripts/manage_cvat.sh restart

# View logs
./scripts/manage_cvat.sh logs

# Prepare sample images
./scripts/manage_cvat.sh samples

# Create backup
./scripts/manage_cvat.sh backup
```

## Creating Your First Annotation Project

### 1. Login to CVAT
1. Open http://localhost:8080 in your browser
2. Login with `admin` / `admin123`

### 2. Create a New Project
1. Click "Projects" in the top menu
2. Click "Create new project"
3. Fill in project details:
   - **Name**: "Trading Card Segmentation"
   - **Labels**: Add a label named "card" with type "polygon"

### 3. Create a Task
1. Inside your project, click "Create new task"
2. Fill in task details:
   - **Name**: "Card Annotation Batch 1"
   - **Subset**: "train" (or "val"/"test")
3. Upload images from `data/sample_images/`
4. Set **Image Quality**: 95
5. Click "Submit & Open"

### 4. Start Annotating
1. Click "Job #1" to start annotation
2. Select the "card" label from the left panel
3. Use the polygon tool to trace card boundaries
4. Follow the annotation guidelines in `docs/CVAT_ANNOTATION_GUIDELINES.md`

## Sample Images

The system includes 5 sample images ready for annotation:
- Located in: `data/sample_images/`
- These are test images from the occlusion test dataset
- Perfect for learning the annotation workflow

## Annotation Guidelines

Detailed annotation guidelines are available in:
`docs/CVAT_ANNOTATION_GUIDELINES.md`

Key points:
- Use polygon tool for precise card boundaries
- Each card gets a separate instance
- Follow card edges precisely (±2 pixel tolerance)
- Include rounded corners properly
- Annotate cards with >70% visibility

## Export Annotations

### COCO Format (Recommended)
1. Go to your project
2. Click "Export dataset"
3. Select "COCO 1.0" format
4. Click "Export"

### YOLO Format (Alternative)
1. Follow same steps as above
2. Select "YOLO 1.1" format instead

## Troubleshooting

### CVAT Not Loading
```bash
# Check service status
./scripts/manage_cvat.sh status

# Restart if needed
./scripts/manage_cvat.sh restart
```

### Upload Issues
- Ensure images are in JPG/PNG format
- Check file sizes (max 50MB per image)
- Verify images are in `data/sample_images/`

### Performance Issues
- Close other browser tabs
- Use Chrome or Firefox for best performance
- Reduce batch size if uploading many images

### Database Issues
```bash
# Create backup before troubleshooting
./scripts/manage_cvat.sh backup

# Restart services
./scripts/manage_cvat.sh restart
```

## File Structure

```
pokemon/
├── cvat_repo/                    # Official CVAT repository
├── data/
│   └── sample_images/           # Images ready for annotation
├── docs/
│   ├── CVAT_ANNOTATION_GUIDELINES.md
│   └── CVAT_SETUP_README.md
├── scripts/
│   └── manage_cvat.sh          # CVAT management script
└── configs/
    └── cvat_project_template.json
```

## Next Steps

1. **Practice Annotation**: Start with the 5 sample images to learn the workflow
2. **Create Guidelines**: Customize annotation guidelines for your specific needs
3. **Scale Up**: Add more images to `data/sample_images/` for larger annotation batches
4. **Quality Control**: Set up review processes for annotation quality
5. **Export Training Data**: Export annotations in COCO format for model training

## Support

- **Annotation Guidelines**: `docs/CVAT_ANNOTATION_GUIDELINES.md`
- **Project Template**: `configs/cvat_project_template.json`
- **Management Script**: `./scripts/manage_cvat.sh help`

## Performance Targets

- **Annotation Speed**: 2-3 minutes per single card image
- **Quality Target**: >95% IoU consistency
- **Batch Size**: 50-100 images per task for optimal performance

---

**Status**: ✅ CVAT is deployed and ready for annotation  
**Last Updated**: September 26, 2025
