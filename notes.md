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

## Training Data Export Feature

### Implementation
Added a "Download Training Data" button to camera_test.html that allows capturing the current model output as training examples for fine-tuning.

### Features
1. **Button**: Enabled when detections are present
2. **Captures**: 
   - Current video frame (without overlay)
   - All detections (not just filtered ones above threshold)
3. **Format**: YOLO OBB format matching training data structure
   - Image: JPG file (95% quality)
   - Labels: Text file with format: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
4. **Naming**: Timestamp-based: `training_YYYY-MM-DD_HH-MM-SS.jpg` and `.txt`

### Implementation Details
- **OBB Conversion**: Converts from (center, width, height, angle) to 4 corner points
- **Coordinate Format**: Normalized [0,1] coordinates
- **Corner Order**: Top-left, top-right, bottom-right, bottom-left
- **Logging**: Full logging of export process and label content
- **Overlay Visualization**: Optional checkbox (checked by default) to also download an annotated version showing bounding boxes and confidence scores for visual verification

### Files Downloaded
When checkbox is enabled (default), downloads 3 files:
1. `training_YYYY-MM-DD_HH-MM-SS.jpg` - Clean image from camera
2. `training_YYYY-MM-DD_HH-MM-SS.txt` - YOLO OBB format labels
3. `training_YYYY-MM-DD_HH-MM-SS_overlay.jpg` - Visualization with bounding boxes and confidence scores

When checkbox is disabled, downloads 2 files (no overlay).

### Use Cases
- Capture real-world examples for model improvement
- Build training dataset from camera feed
- Fine-tune model with edge cases or misdetections
- Quick data collection without separate annotation tools
- Visual verification of detections via overlay image

## Frontend Development Volume Mount

### Setup
Updated docker-compose.yml to mount the entire `frontend/public` directory into the container:
- Mount: `./frontend/public:/usr/share/nginx/html`
- This replaces the previous models-only mount

### Benefits
- Changes to any file in `frontend/public` are immediately reflected
- No need to rebuild container for HTML/JS/CSS changes
- Faster development iteration
- Works for camera_test.html, models, onnx files, etc.

### Usage
```bash
# Recreate the container with new volume mount
docker compose up -d --force-recreate frontend
```

After this, any edits to files in `frontend/public/` on the host machine will immediately be served by nginx in the container.

## Object Tracking Implementation (ByteTrack)

### Overview
Implemented ByteTrack object tracking algorithm to stabilize detections and reduce flickering in camera_test.html. The tracking system maintains stable IDs across frames and filters out low-confidence false positives.

### Files Created
- **bytetrack.js**: Standalone JavaScript module containing:
  - `KalmanFilter`: 2D position and velocity tracking with constant velocity model
  - `Track`: Individual tracked object with state and history
  - `ByteTracker`: Main tracking algorithm with two-stage association

### Integration in camera_test.html
1. **Script Import**: Added `<script src="bytetrack.js"></script>`
2. **Feature Flag**: Added checkbox "Enable Object Tracking (ByteTrack)"
3. **State Variables**:
   - `trackingEnabled`: Boolean flag for tracking on/off
   - `tracker`: ByteTracker instance (null when disabled)

### Behavior

#### When Disabled (Default)
- Shows raw model outputs
- All detections above minimum threshold (0.3) are displayed
- Filtering only by confidence threshold slider
- Labels show: `{confidence}%`
- This is the original behavior

#### When Enabled
- Raw detections are passed through ByteTracker
- Only confirmed tracks (3+ hits) are displayed
- Two-stage association with high/low confidence thresholds
- Kalman filter predictions smooth position across frames
- Labels show: `ID:{trackId} {confidence}%`
- Stable IDs persist across frames

### ByteTracker Parameters
```javascript
trackHighThresh: 0.6,    // High confidence threshold
trackLowThresh: 0.3,     // Low confidence threshold  
newTrackThresh: 0.7,     // Threshold to start new track
matchThresh: 0.8,        // IoU threshold for matching
maxAge: 30,              // Frames to keep without detection
minHits: 3               // Min detections to confirm track
```

### How ByteTrack Works
1. **Prediction**: All active tracks predict their next position using Kalman filter
2. **First Association**: High-confidence detections matched to tracks via IoU
3. **Second Association**: Low-confidence detections matched to unmatched tracks
4. **Track Creation**: New tracks created from high-confidence unmatched detections
5. **Track Deletion**: Tracks removed after `maxAge` frames without matches
6. **Confirmation**: Only tracks with `minHits` or more detections are returned

### Benefits
- **Reduced Flickering**: Kalman filter smooths position predictions
- **Stable IDs**: Objects maintain same ID across frames
- **False Positive Filtering**: Requires multiple hits before confirmation
- **Lost Detection Recovery**: Can match low-confidence detections to existing tracks
- **Handles Occlusion**: Keeps tracks alive for up to 30 frames without detection

### Use Cases
- **Auto Mode**: Stabilizes continuous detection stream
- **Training Data Export**: Track IDs included in overlay visualizations
- **Object Counting**: Stable IDs enable accurate counting
- **Quality Control**: Only confirmed tracks shown, reducing noise

### Testing
1. Enable "Start Auto" mode
2. Check "Enable Object Tracking (ByteTrack)"
3. Observe stable IDs appearing on detected cards after 3 frames
4. Notice smoother box positions compared to raw detections
5. Cards will maintain same ID even if temporarily occluded

### Parameter Controls

Added interactive sliders for all ByteTrack parameters with live tuning:

**UI Features:**
- Parameters panel appears when tracking is enabled
- All parameters update in real-time
- Tracker reinitializes automatically when parameters change
- Each parameter includes description explaining its purpose

**Tunable Parameters:**

1. **High Confidence Threshold (30-95%)**: Default 60%
   - Detections above this are prioritized for first-stage matching
   - Higher = more selective initial matching

2. **Low Confidence Threshold (10-60%)**: Default 30%
   - Minimum confidence for second-stage recovery matching
   - Helps recover temporarily low-confidence detections

3. **New Track Threshold (40-95%)**: Default 70%
   - Minimum confidence to start tracking new objects
   - Higher = prevents false positives from creating tracks

4. **Match Threshold IoU (30-95%)**: Default 80%
   - Minimum Intersection-over-Union for detection-to-track matching
   - Higher = stricter spatial matching

5. **Max Age (5-60 frames)**: Default 30
   - How long tracks persist without matched detections
   - Higher = more tolerant of occlusion/missed detections

6. **Minimum Hits (1-10)**: Default 3
   - Consecutive detections required before track confirmation
   - Higher = fewer false positives, slower initial detection

**Behavior:**
- Changing parameters while tracking is enabled reinitializes the tracker
- Previous tracks are cleared when parameters change
- Parameters persist even when tracking is disabled/re-enabled
- All changes are logged for debugging

**Use Cases:**
- **High false positive rate**: Increase newTrackThresh and minHits
- **Missing detections**: Lower trackLowThresh, increase maxAge
- **Identity switching**: Increase matchThresh (stricter spatial matching)
- **Slow to detect**: Lower minHits (faster but more false positives)
- **Cards moving fast**: Lower matchThresh (more lenient matching)

