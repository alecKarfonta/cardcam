# Model Setup Instructions

## Required Files Not in Git

The following files are required for the GPU acceleration tests but are excluded from git due to size:

### ONNX Model File
- **File**: `trading_card_detector.onnx` (11MB)
- **Source**: Copy from `frontend/public/models/trading_card_detector.onnx`
- **Command**: `cp ../frontend/public/models/trading_card_detector.onnx ./`

### Setup Commands

To set up the complete test environment:

```bash
cd gpu-acceleration-tests

# Copy the ONNX model (required)
cp ../frontend/public/models/trading_card_detector.onnx ./

# Verify files are present
ls -la trading_card_detector.onnx mtg.png

# Build and run with Docker
docker-compose up --build

# Or build manually
docker build -t gpu-acceleration-tests .
docker run -d -p 8081:8080 --name gpu-acceleration-tests gpu-acceleration-tests
```

### File Verification

After setup, you should have:
- ✅ `trading_card_detector.onnx` (11,078,355 bytes)
- ✅ `mtg.png` (617,005 bytes) 
- ✅ `onnx/` directory with ONNX Runtime files
- ✅ All JavaScript test modules

### Access

Open http://localhost:8081 to access the test suite.

## Test Sequence

1. **WASM Baseline**: Test with optimized multi-threaded WASM
2. **WebGPU Test**: Attempt GPU acceleration (may cause browser lockup)
3. **WebNN Test**: Test WebNN execution provider integration
4. **Full Benchmark**: Compare all methods systematically

The test suite will systematically debug GPU acceleration issues and provide detailed performance analysis.
