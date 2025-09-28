# Trading Card Scanner Debug Notes

## Current Issues (2025-09-28)

### 1. ONNX Model Loading Error
- **Error**: `ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because protobuf parsing failed`
- **Model**: `/models/trading_card_detector_1080.onnx`
- **File Size**: 10,706,221 bytes (10.7MB)
- **File Type**: `data` (generic binary)

### 2. Configuration Mismatch
- **model_info.json**: `input_size: 640`
- **modelConfig.ts**: `inputSize: 1080`
- **Potential Issue**: Model was trained for 640px input but config expects 1080px

### 3. WebGL Backend Issues
- **Warning**: `removing requested execution provider "webgl" from session options because it is not available: backend not found`
- **Fallback**: Using WASM backend only

### 4. WebSocket Connection Failed
- **Error**: `WebSocket connection to 'wss://mlapi.us:3001/ws' failed`
- **Note**: This is secondary to the model loading issue

## Available Model Files
- `trading_card_detector_1080.onnx` (10.7MB) - **CURRENT/FAILING**
- `trading_card_detector_1088_final.onnx` (11.0MB)
- `trading_card_detector_1088_v2.onnx` (11.0MB) 
- `trading_card_detector_1088.onnx` (11.0MB)
- `trading_card_detector_backbone.onnx` (10.8MB)
- `trading_card_detector.onnx` (11.0MB)

## Debugging Steps Tried
1. ✅ Verified model file exists and has reasonable size
2. ❌ Python ONNX validation (onnx module not available)
3. ✅ Fixed input size configuration mismatch (1080 → 640px)
4. ✅ Changed model file from trading_card_detector_1080.onnx to trading_card_detector.onnx
5. 🔄 Testing if model loads correctly with fixed configuration

## Configuration Changes Made
- **Model Path**: `/models/trading_card_detector_1080.onnx` → `/models/trading_card_detector.onnx`
- **Input Size**: `1080` → `640` (matches model_info.json)
- **NMS Config**: Updated all inputSize references to 640

## Additional Fixes Applied
- **ONNX Runtime Setup**: Enabled SIMD for better performance
- **WebGL Detection**: Added WebGL context detection and logging
- **Development Server**: Started and confirmed running on localhost:3000

## Next Steps
1. ✅ Test if model loads correctly with fixed configuration
2. ✅ Address WebGL backend configuration if needed  
3. 🔄 Fix WebSocket connection after model loading works
4. 🔄 Verify camera functionality works end-to-end

## New Findings (After npm installation)
- **Node.js/npm**: ✅ Installed via nvm (Node v24.9.0, npm v11.6.0)
- **Model File Analysis**: ✅ All ONNX files have valid headers and are not corrupted
- **Model Origin**: Models created with PyTorch 2.10.0.dev20250926+cu128 (very recent dev version)
- **Potential Issue**: Opset version compatibility between recent PyTorch export and ONNX Runtime 1.21.0

## Solutions Attempted
1. **Configuration Fixes**: ✅ Fixed input size mismatch (1080→640px)
2. **Model Path Changes**: ✅ Tried multiple model files
3. **ONNX Runtime Updates**: ✅ Disabled graph optimization, enabled verbose logging
4. **Execution Providers**: ✅ Switched to WASM-only for compatibility
5. **Debug Tool**: ✅ Created `/debug_onnx.html` to test all models

## Latest Updates (2025-09-28)
- **TypeScript Errors**: ✅ Fixed array type issues in ModelManager and OBBNMSProcessor
- **Input Size**: ✅ Corrected to 1088px (matches training)
- **Model Path**: ✅ Using trading_card_detector_1088_final.onnx
- **Development Server**: ✅ Running on http://localhost:3002/cardcam
- **ONNX Runtime Upgrade**: ❌ Blocked by dependency conflicts (@types/node)

## 🎯 PRIMARY GOAL: Enable Newer Opset Support for OBB Decode Layers ✅ ACHIEVED

### CRITICAL BUG FIX (2025-09-28 14:00):
**Issue**: Detection results were completely wrong - detecting 3 cards when only 1 visible, all clustered at left edge
**Root Cause**: Model output format mismatch and incorrect processing
**Fixes Applied**:
1. **Dynamic Anchor Count**: Model outputs 24,276 anchors (not 8,400 as expected)
   - Fixed NMS processor to calculate `numAnchors = rawOutput.length / numChannels`
2. **Angle Processing**: Raw angles in [0,1] range, not radians
   - Fixed conversion: `angle = angleRaw * 2 * Math.PI`
3. **Applied to both standalone test and frontend code**

**Result**: Now correctly detects single Pokemon card with 83.2% confidence at proper location!
**CRITICAL**: We need newer opset (18+) working to properly export final decode layers for oriented bounding boxes. Current opset 11 downgrade loses essential OBB functionality.

### Why Newer Opset is Essential:
- **Oriented Bounding Box Decode Layers**: Require opset 17+ operators
- **Complete Pipeline Export**: Need full model with decode layers, not just backbone
- **Angle Processing**: OBB angle decode operations need newer ONNX operators
- **NMS for OBB**: Oriented NMS requires advanced operators only in newer opsets

## 🎉 BREAKTHROUGH: Opset 18 + OBB Decode Layers Working!

### ✅ Successfully Achieved:
- **ONNX Runtime**: ✅ UPGRADED to 1.23.0 (supports opset 18)
- **Model Export**: ✅ Complete OBB backbone with decode layers (opset 18)
- **Output Format**: ✅ Raw (1, 6, 24276) - [cx, cy, w, h, angle, confidence]
- **Decode Layers**: ✅ Angle processing and coordinate decode included
- **File Size**: ✅ 10.5 MB optimized model

### ⚠️ NMS Integration Status:
- **Integrated NMS**: ❌ Blocked by PyTorch export limitations (dynamic shapes)
- **JavaScript NMS**: ✅ Required for post-processing (already implemented)
- **Workaround**: Export backbone + use existing JavaScript NMS pipeline

## Status
- **Production Server**: ✅ Running in Docker on http://localhost:3001
- **Container**: ✅ card-scanner-frontend (nginx + React build)
- **Model Configuration**: ✅ Using trading_card_detector_backbone_opset18.onnx (COMPLETE OBB)
- **ONNX Runtime**: ✅ Version 1.23.0 with opset 18 support
- **Debug Tool**: ✅ Available at http://localhost:3001/debug_onnx.html
- **Root Cause**: RESOLVED - Now have complete OBB pipeline with modern opset

## Network Architecture

### Connection Flow
```
Internet → mlapi.us (192.168.1.196) → nginx proxy → 192.168.1.77:3001 → Docker container
```

### External Nginx Proxy (192.168.1.196)
- **Domain**: mlapi.us (SSL with Let's Encrypt)
- **Proxy Rules**:
  - `/cardcam/static/` → `http://192.168.1.77:3001/static/`
  - `/cardcam/onnx/` → `http://192.168.1.77:3001/onnx/`
  - `/cardcam/models/` → `http://192.168.1.77:3001/models/`
  - `/cardcam/` → `http://192.168.1.77:3001/cardcam/`

### Local Docker Container (192.168.1.77:3001)
- **Container**: card-scanner-frontend
- **Nginx Config**: Serves React app at `/cardcam/` with WebAssembly support
- **Static Assets**: Available at root paths (`/static/`, `/onnx/`, `/models/`)

## Latest Features (2025-09-28)
- **Docker Deployment**: ✅ Production-ready containerized frontend
  - Multi-stage Docker build (Node.js build + nginx serve)
  - Optimized nginx configuration with security headers
  - ONNX model file serving with proper CORS headers
  - WebAssembly support with 'unsafe-eval' CSP directive
  - Running on port 3001 via docker-compose
  - Production build with gzip compression

- **Confidence Slider**: ✅ Added dynamic confidence threshold filter
  - Real-time adjustment from 0% to 100% (step: 5%)
  - Client-side filtering in DetectionOverlay component
  - Shows filtered count vs total count in debug status
  - Styled to match camera interface design
  - Mobile-responsive with touch-friendly controls
  - Works with any inference backend (no model changes needed)

## Troubleshooting

### WebAssembly CSP Issues
If you see `WebAssembly.instantiate(): Refused to compile` errors:

**External Nginx Proxy CSP** (192.168.1.196) must include:
```nginx
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' wss: ws:; media-src 'self'; object-src 'none'; frame-ancestors 'self'; worker-src 'self' blob:; child-src 'self' blob:;" always;
```

**Required CSP Directives for ONNX Runtime:**
- `'unsafe-eval'` - Required for WebAssembly compilation
- `worker-src 'self' blob:` - For Web Workers used by ONNX Runtime
- `child-src 'self' blob:` - For child contexts

### Model Loading Issues
1. Check network tab for 404s on `/cardcam/models/` paths
2. Verify external nginx proxy routes are correct
3. Check console for CSP violations
4. Test direct access: `http://192.168.1.77:3001/models/trading_card_detector.onnx`
