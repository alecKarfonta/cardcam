## Current Issue: ONNX Runtime MIME Type Error in Production

### Problem
- ONNX Runtime WebGPU fails to load in production
- Error: `.mjs` files served with wrong MIME type "application/octet-stream" instead of "text/javascript"
- This breaks ES module imports for WebAssembly

### Root Cause
- nginx default MIME types don't include `.mjs` extension
- nginx.conf had `.mjs` location block but it wasn't being used due to ordering

### Solution
- Added explicit MIME type mapping in frontend/nginx.conf:
  ```
  types {
      text/javascript mjs;
  }
  ```

### Resolution - COMPLETED
- Rebuilt frontend production container with updated nginx.conf
- Recreated container with `docker compose up -d --force-recreate frontend`
- Verified ONNX files are accessible in /onnx/ directory
- Tested MIME type serving:
  - Direct container: Content-Type: text/javascript ✓
  - Through proxy: Content-Type: text/javascript ✓

### Test
camera_test.html should now load successfully at https://mlapi.us/cardcam/camera_test.html

## WebGPU-Only Execution

### Changes Made
- Removed WASM fallback from camera_test.html
- Changed `executionProviders: ['webgpu', 'wasm']` to `executionProviders: ['webgpu']`
- Updated provider display to show only "WebGPU"
- Added full error stack logging for debugging WebGPU failures
- No fallback means WebGPU errors will be visible instead of silently falling back to WASM

### Why
- User wants to see actual WebGPU failures to debug properly
- No silent fallbacks that hide the root cause
- Forces WebGPU to work or fail explicitly

## Cross-Origin Isolation Headers for WASM/WebGPU

### Issue Found
- WebGPU backend actually USES WebAssembly under the hood (it's not pure GPU)
- ONNX Runtime WebGPU requires SharedArrayBuffer which needs cross-origin isolation
- `.mjs` and `.wasm` files need Cross-Origin-Embedder-Policy and Cross-Origin-Opener-Policy headers

### Frontend Container Fix (DONE)
- Added cross-origin headers to `.mjs` and `.wasm` location blocks in frontend/nginx.conf
- Removed `.mjs` from general static assets block to prevent header conflicts
- Headers now set:
  - Cross-Origin-Embedder-Policy: require-corp
  - Cross-Origin-Opener-Policy: same-origin
  - Content-Type: text/javascript (for .mjs)
  - Cache-Control: public, immutable

### Production Proxy Fix (NEEDS DEPLOYMENT)
- Updated production_server_nginx to add cross-origin headers to /cardcam/ location
- This file needs to be deployed to 192.168.1.196 and nginx reloaded:
  ```bash
  # On 192.168.1.196:
  sudo cp production_server_nginx /etc/nginx/sites-available/mlapi.us
  sudo nginx -t
  sudo systemctl reload nginx
  ```

## Model Output Issue - camera_test.html

### Problem
- Runtime error: "Cannot read properties of undefined (reading 'data')"
- At line 301: `results[session.outputNames[0]].data`
- Model inference runs but doesn't return expected outputs

### Solution - IMPLEMENTED
Implemented comprehensive logging system with proper error handling:

#### Logging System Features
- **Log Levels**: DEBUG, INFO, WARNING, ERROR with color coding
- **Format**: `HH:MM:SS [ModuleName.functionName] message`
- **Separate Loggers**: Camera, Model, Inference, Render, UI, Main
- **Structured Data**: Optional data objects for complex information

#### Logging Coverage
1. **Camera Module**: 
   - Camera initialization, permissions, video metadata
   - Frame capture and dimensions
   
2. **Model Module**:
   - ONNX Runtime version and availability
   - Model loading time and configuration
   - Input/output names and types
   
3. **Inference Module**:
   - Frame capture details
   - Preprocessing steps (scaling, padding, tensor creation)
   - Inference timing
   - Output validation and tensor shapes
   - Detection processing and confidences
   
4. **Render Module**:
   - Detection counts
   - Threshold filtering
   - Bounding box rendering
   
5. **UI Module**:
   - Button clicks and user interactions
   - Threshold changes
   - Stats updates
   
6. **Main Module**:
   - Application startup
   - Browser capabilities (WebGPU support)

#### Benefits
- Easy to identify which module/function is executing
- Detailed data about model inputs/outputs for debugging
- Color-coded console output for quick visual scanning
- Can adjust log level per module if needed
- All error conditions properly logged with context

### Diagnosis - FOUND ROOT CAUSE
The logs revealed the issue:
- **Model has only 1 output** named "output0"
- **Code expects 2 outputs**: boxes and scores separately
- Original code tried to access `session.outputNames[1]` which is undefined
- This caused the validation check to fail

### Solution - FIXED
**Model Output Format:**
- Shape: `[1, 300, 7]`
  - Batch: 1
  - Max detections: 300
  - Values per detection: 7
  
**Format: Oriented Bounding Boxes (OBB)**
Each detection contains: `[x_center, y_center, width, height, confidence, class_id, angle]`
- x, y: Center coordinates in pixels (1088x1088 space)
- w, h: Box dimensions in pixels
- confidence: Detection confidence score
- class_id: Class identifier (0 for trading cards)
- angle: Rotation angle in radians

**Code Changes:**
1. **Detection Parsing**: 
   - Parse single output tensor with 7 values per detection
   - Normalize coordinates from 1088x1088 to [0,1]
   - Extract rotation angle for each detection
   
2. **Rendering**: 
   - Use canvas transforms to draw rotated rectangles
   - Save/restore context for each box
   - Translate to center, rotate, draw rect centered at origin
   - Keep confidence labels unrotated for readability
   
3. **Logging**: 
   - Log output format (OBB)
   - Log number of detections found
   - Log detection confidences

### Status - READY TO TEST
The camera_test.html now:
- Correctly parses OBB format output
- Draws rotated bounding boxes on video
- Filters by confidence threshold
- Logs all detection details

Reload the page and test with trading cards at various angles.

