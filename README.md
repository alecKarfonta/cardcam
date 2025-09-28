# Trading Card Scanner 🃏

A real-time trading card detection and extraction system using YOLO11 Oriented Bounding Box (OBB) models. Capable of detecting and extracting individual cards from complex multi-card scenes with high accuracy.

## 🚀 Live Demo

**Web Application**: [https://mlapi.us/cardcam/](https://mlapi.us/cardcam/)

Try the live camera-based card scanner directly in your browser! Works on both desktop and mobile devices.

## Project Overview

This system provides an end-to-end pipeline for detecting and extracting trading cards from various image inputs with **99.5%+ mAP@0.5** accuracy for real-world deployment scenarios.

### Key Features

- **🎯 High Accuracy**: 99.5% mAP@0.5 on validation set
- **⚡ Real-time Processing**: <60ms inference time per image
- **📱 Cross-Platform**: Web application works on desktop and mobile
- **🔄 Oriented Detection**: Handles rotated cards with precise angle detection
- **📦 Card Extraction**: Automatic cropping and individual card extraction
- **🎮 Multi-Card Support**: Detects multiple cards in complex scenes

### Supported Card Types

- Magic: The Gathering
- Pokémon TCG
- Yu-Gi-Oh!
- Sports cards (Baseball, Basketball, etc.)
- Other trading card games

## 📊 Model Performance

### Current Model: YOLO11n-OBB v15

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 99.5% |
| **mAP@0.5-0.95** | 99.4% |
| **Precision** | 99.9% |
| **Recall** | 99.9% |
| **Inference Time** | ~54ms |
| **Model Size** | 11.0 MB |

### Training Results

![Training Progress](src/training/trading_cards_obb/yolo11n_obb_v15/results.png)
*Training curves showing loss and accuracy metrics over epochs*

![Confusion Matrix](src/training/trading_cards_obb/yolo11n_obb_v15/confusion_matrix_normalized.png)
*Normalized confusion matrix showing classification performance*

## 🎯 Example Outputs

### Single Card Detection
```
Input: Single Pokemon card image
Output: 
- Bounding box: [x, y, width, height, angle]
- Confidence: 0.95
- Processing time: 45ms
```

![Single Card Example](outputs/gold_visualizations/individual_20250927_111206.jpg)
*Example: Single card detection with oriented bounding box*

### Multi-Card Scene Detection
```
Input: Multiple cards in scene
Output: 
- Card 1: confidence=0.92, angle=15°
- Card 2: confidence=0.89, angle=-5°  
- Card 3: confidence=0.94, angle=0°
- Total processing time: 58ms
```

![Multi-Card Example](src/training/trading_cards_obb/yolo11n_obb_v15/val_batch0_pred.jpg)
*Example: Multi-card detection with individual bounding boxes*

### Validation Batch Results
![Validation Labels](src/training/trading_cards_obb/yolo11n_obb_v15/val_batch0_labels.jpg)
*Ground truth labels*

![Validation Predictions](src/training/trading_cards_obb/yolo11n_obb_v15/val_batch0_pred.jpg)
*Model predictions on validation set*

## Technical Stack

- **Deep Learning**: PyTorch 2.3.1 + Ultralytics YOLO11
- **Frontend**: React + TypeScript + ONNX Runtime Web
- **Backend**: Python + FastAPI + ONNX Runtime
- **Deployment**: Docker + Nginx
- **Model Format**: ONNX (optimized for web deployment)


## 🚀 Getting Started

### Web Application (Recommended)

The easiest way to try the system is through the web application:

1. **Visit**: [https://mlapi.us/cardcam/](https://mlapi.us/cardcam/)
2. **Allow camera access** when prompted
3. **Point camera at trading cards** and see real-time detection
4. **Capture and extract** individual cards with the capture button

### Local Development Setup

#### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend)
- Docker and Docker Compose
- CUDA-capable GPU (optional, for training)

#### Frontend Development

```bash
# Clone the repository
git clone https://github.com/alecKarfonta/cardcam.git
cd pokemon/frontend

# Install dependencies
npm install

# Start development server
npm start
# Opens http://localhost:3000
```

#### Backend/Training Setup

```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training (optional)
python src/training/train_yolo_obb.py
```

#### Docker Deployment

```bash
# Build and run the complete system
docker-compose up -d

# Access the application
open http://localhost:3001/cardcam/
```

### API Usage

```python
import requests
import base64

# Encode image
with open("card_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post("https://mlapi.us/api/detect", json={
    "image": image_data,
    "confidence_threshold": 0.8
})

results = response.json()
print(f"Detected {len(results['detections'])} cards")

# Each detection contains:
# - bbox: [x, y, width, height, angle]
# - confidence: float
# - extracted_image: base64 encoded cropped card
```

## 📁 Project Structure

```
pokemon/
├── frontend/              # React web application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── utils/         # ONNX model managers
│   │   └── store/         # Redux state management
│   ├── public/models/     # ONNX model files
│   └── nginx.conf         # Production nginx config
├── src/
│   ├── training/          # Model training scripts
│   │   └── trading_cards_obb/  # Training results
│   ├── data/              # Data processing utilities
│   └── utils/             # Helper functions
├── data/                  # Training datasets
├── outputs/               # Training outputs and visualizations
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── docker-compose.yml     # Container orchestration
```

## 🔧 Model Architecture

### YOLO11n-OBB (Oriented Bounding Box)

- **Base Model**: YOLOv11 Nano with OBB head
- **Input Size**: 1088×1088 pixels
- **Output Format**: `[cx, cy, w, h, angle, confidence, class]`
- **Angle Range**: -90° to +90° (normalized to [0,1])
- **Classes**: Single class ("trading_card")

### Key Improvements

1. **Oriented Detection**: Handles rotated cards accurately
2. **Lightweight**: 11MB model size for web deployment
3. **Fast Inference**: Optimized for real-time processing
4. **High Precision**: 99.9% precision on validation set

## 🎮 Features

### Web Application Features

- **📷 Real-time Camera**: Live card detection through webcam
- **🎯 Confidence Filtering**: Adjustable detection threshold
- **📦 Card Extraction**: Automatic cropping of detected cards
- **💾 Batch Download**: Export all detected cards as images
- **📱 Mobile Support**: Works on smartphones and tablets
- **🔍 Zoom Controls**: Detailed card inspection (50%-400% zoom)

### Technical Features

- **⚡ ONNX Runtime**: Optimized inference in browser
- **🔄 WebGL Acceleration**: GPU acceleration when available
- **📊 Real-time Metrics**: Processing time and detection counts
- **🎨 Visual Overlays**: Bounding box visualization with confidence scores
- **🔧 Debug Tools**: Built-in model testing and validation tools

## 🚀 Development Status

✅ **Phase 1 Complete**: Foundation & Data Collection  
✅ **Phase 2 Complete**: Model Training & Optimization  
✅ **Phase 3 Complete**: Web Application Development  
✅ **Phase 4 Complete**: Production Deployment  
🔄 **Phase 5 Current**: Continuous Improvement & Feature Enhancement

### Recent Achievements

- 🎯 Achieved 99.5% mAP@0.5 on validation set
- ⚡ Reduced inference time to <60ms per image
- 📱 Deployed production web application
- 🔧 Implemented card extraction and batch download
- 📊 Added comprehensive performance monitoring

## 📈 Performance Benchmarks

### Inference Speed Comparison

| Platform | Device | Inference Time | FPS |
|----------|--------|----------------|-----|
| Web (Chrome) | Desktop CPU | ~54ms | ~18 FPS |
| Web (Chrome) | Desktop GPU | ~45ms | ~22 FPS |
| Web (Safari) | iPhone 14 | ~78ms | ~13 FPS |
| Python | RTX 5090 | ~12ms | ~83 FPS |
| Python | CPU | ~89ms | ~11 FPS |

### Accuracy by Card Type

| Card Type | mAP@0.5 | Sample Size |
|-----------|---------|-------------|
| Pokemon | 99.8% | 1,200 images |
| Magic: The Gathering | 99.4% | 800 images |
| Yu-Gi-Oh! | 99.2% | 600 images |
| Sports Cards | 99.6% | 400 images |

## 🛠️ Troubleshooting

### Common Issues

**Camera not working?**
- Ensure HTTPS connection (required for camera access)
- Check browser permissions for camera access
- Try refreshing the page

**Model not loading?**
- Check network connection
- Verify WebAssembly support in browser
- Try disabling ad blockers

**Poor detection accuracy?**
- Ensure good lighting conditions
- Keep cards flat and unobstructed
- Adjust confidence threshold slider

### Browser Compatibility

| Browser | Desktop | Mobile | WebGL | Performance |
|---------|---------|--------|-------|-------------|
| Chrome | ✅ | ✅ | ✅ | Excellent |
| Firefox | ✅ | ✅ | ✅ | Good |
| Safari | ✅ | ✅ | ⚠️ | Good |
| Edge | ✅ | ✅ | ✅ | Excellent |

## 📚 Documentation

- **[Frontend Roadmap](docs/FRONTEND_ROADMAP.md)**: Web application development plan
- **[Training Guide](docs/YOLO_OBB_TRAINING.md)**: Model training instructions
- **[CVAT Setup](docs/CVAT_SETUP_README.md)**: Annotation platform setup
- **[Development Roadmap](developemnt-roadmap.md)**: Complete project roadmap

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLO11 implementation
- **ONNX Runtime**: For enabling web deployment
- **React Team**: For the robust frontend framework
- **Trading Card Communities**: For providing valuable feedback and testing

## 📞 Contact

- **Project Lead**: [Alec Karfonta](https://github.com/alecKarfonta)
- **Live Demo**: [https://mlapi.us/cardcam/](https://mlapi.us/cardcam/)
- **Issues**: [GitHub Issues](https://github.com/alecKarfonta/cardcam/issues)

---

⭐ **Star this repository if you find it useful!**

*Built with ❤️ for the trading card community*
