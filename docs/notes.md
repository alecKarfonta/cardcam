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

### Investigation
- Added debug logging to see:
  - session.outputNames (what outputs the model claims to have)
  - Object.keys(results) (what outputs are actually returned)
  - Full results object
- Added proper error handling with descriptive message

### Possible Causes
1. Model output names don't match what we expect
2. WebGPU inference failing silently and returning empty/wrong outputs
3. Model file not compatible with WebGPU execution provider
4. Model might need different output handling for WebGPU vs WASM

### Next Steps
- Check browser console for the debug logs to see what outputs are actually returned
- Verify model file is correct and compatible with ONNX Runtime WebGPU

