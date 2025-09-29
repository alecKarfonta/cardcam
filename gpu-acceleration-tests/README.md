# GPU Acceleration Test Suite

A comprehensive standalone testing environment for browser-based GPU acceleration technologies including WebGPU, WebNN, and optimized WebAssembly.

## Overview

This test suite was created to systematically explore and benchmark GPU acceleration options in web browsers for machine learning inference. It provides isolated testing of different acceleration methods with detailed performance analysis and stability testing.

## Features

### System Information Detection
- Browser and GPU detection
- CPU core count and memory information
- WebGPU and WebNN support detection
- Cross-origin isolation status

### WebGPU Testing
- Basic WebGPU API functionality
- Compute shader performance testing
- ONNX Runtime WebGPU integration testing
- Non-blocking execution approaches
- Error handling and stability analysis

### WebNN Testing
- Browser WebNN API exploration
- Context creation and operation support
- ONNX Runtime WebNN integration
- Performance benchmarking
- Compatibility analysis

### WASM Baseline Testing
- Single-threaded WASM performance
- Multi-threaded WASM with threading
- SIMD acceleration testing
- Cross-origin isolation requirements

### Comprehensive Benchmarking
- Full system performance analysis
- Input size scaling tests
- Memory usage patterns
- Concurrency testing
- Stability analysis over multiple runs
- Performance comparison and recommendations

## Quick Start

### Using Docker (Recommended)

1. Build and run the container:
```bash
cd /home/alec/git/pokemon/gpu-acceleration-tests
docker-compose up --build
```

2. Open your browser to: `http://localhost:8080`

### Manual Setup

1. Serve the files with a web server that supports the required headers:
```bash
# Using Python
python -m http.server 8080

# Using Node.js serve
npx serve -p 8080 -C

# Using nginx (copy nginx.conf to your nginx config)
nginx -c /path/to/nginx.conf
```

2. **Important**: For full functionality, you need cross-origin isolation headers:
   - `Cross-Origin-Embedder-Policy: require-corp`
   - `Cross-Origin-Opener-Policy: same-origin`

## Test Modules

### 1. System Information (`system-info.js`)
Detects and displays:
- Browser type and version
- GPU information (via WebGL debug info)
- CPU cores and memory
- WebGPU/WebNN support status
- Cross-origin isolation status

### 2. WebGPU Tests (`webgpu-tests.js`)
- **Basic Test**: WebGPU adapter/device creation
- **Compute Shader Test**: GPU compute performance with real workloads
- **ONNX Integration Test**: WebGPU execution provider testing

### 3. WebNN Tests (`webnn-tests.js`)
- **Basic API Test**: navigator.ml availability and context creation
- **Context Deep Dive**: Operation support and capabilities
- **ONNX Integration**: WebNN execution provider compatibility

### 4. WASM Tests (`wasm-tests.js`)
- **Single-threaded**: Baseline CPU performance
- **Multi-threaded**: Threading performance with cross-origin isolation
- **SIMD**: Vectorized operations testing

### 5. Test Model (`test-model.js`)
- Synthetic model generation for consistent testing
- Multiple input size testing
- Simulated inference for different execution providers
- Performance scaling analysis

### 6. Benchmark Suite (`benchmark.js`)
- Comprehensive performance analysis
- Stability testing over multiple runs
- Memory usage patterns
- Concurrency analysis
- Final recommendations and best performer identification

## Understanding the Results

### Performance Metrics
- **Execution Time**: Time to complete inference (lower is better)
- **Throughput**: Operations per second (higher is better)
- **Stability**: Consistency across multiple runs
- **Memory Usage**: Peak and baseline memory consumption

### Status Indicators
- ✅ **Green**: Feature working correctly
- ⚠️ **Yellow**: Feature available but with limitations
- ❌ **Red**: Feature not supported or failing

### Common Issues and Solutions

#### WebGPU Browser Lockups
**Problem**: Browser freezes during WebGPU compute operations
**Solution**: The test suite uses non-blocking approaches with timeouts and requestAnimationFrame

#### Cross-Origin Isolation Required
**Problem**: Multi-threading and some GPU features require special headers
**Solution**: Use the provided nginx.conf or Docker setup

#### ONNX Runtime Not Found
**Problem**: Tests fail because ONNX Runtime is not loaded
**Solution**: The tests will attempt to load ONNX Runtime from `/onnx/` path

## Key Findings from Development

Based on extensive testing documented in `../notes.md`:

### WebGPU Status: **PROBLEMATIC**
- Causes browser lockups during `session.run()`
- Synchronous operations block main thread
- Not production-ready for ML inference

### WebNN Status: **INCOMPLETE**
- Browser API works but ONNX Runtime integration fails
- Limited data type support (no int64)
- Inconsistent implementation across browsers

### WASM Status: **RECOMMENDED**
- Reliable and stable performance
- Multi-threading provides significant speedup
- SIMD acceleration available
- No browser lockups or stability issues

## Architecture

```
gpu-acceleration-tests/
├── index.html              # Main test interface
├── system-info.js          # System detection
├── webgpu-tests.js         # WebGPU testing
├── webnn-tests.js          # WebNN testing  
├── wasm-tests.js           # WASM baseline testing
├── test-model.js           # Synthetic model generation
├── benchmark.js            # Comprehensive benchmarking
├── nginx.conf              # Server configuration
├── Dockerfile              # Container setup
├── docker-compose.yml      # Easy deployment
└── README.md               # This file
```

## Browser Compatibility

### WebGPU Support
- Chrome 113+: ✅ Supported (but may cause lockups)
- Firefox 110+: ⚠️ Behind flag
- Safari 16.4+: ⚠️ Experimental
- Edge 113+: ✅ Supported (but may cause lockups)

### WebNN Support
- Chrome 116+: ⚠️ Behind flag
- Firefox: ❌ Not supported
- Safari: ❌ Not supported
- Edge 116+: ⚠️ Behind flag

### WASM Threading Support
- All modern browsers: ✅ Supported with cross-origin isolation

## Development Notes

This test suite was developed after extensive attempts to get GPU acceleration working in production. The key insights:

1. **WebGPU is not ready for production ML inference** due to blocking behavior
2. **WebNN browser support exists but ONNX Runtime integration is broken**
3. **Optimized multi-threaded WASM is currently the best solution**

See `../notes.md` for detailed analysis of each approach attempted.

## Contributing

To add new tests or improve existing ones:

1. Follow the existing module pattern
2. Include comprehensive error handling
3. Add progress indicators and detailed logging
4. Test across multiple browsers
5. Update this README with new findings

## License

This test suite is part of the Pokemon card detection project and follows the same license terms.
