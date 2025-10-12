# WebGPU Model Inference - Complete Fix Plan

## Problem Summary

After days of struggling, we're facing these core issues:
1. Model either doesn't load or freezes during inference
2. Falling back to WASM even though WebGPU should work
3. Cross-origin isolation not enabled (crossOriginIsolated: false)
4. Browser lockups during `session.run()`

## Root Cause Analysis

### Issue 1: Wrong ONNX Runtime Import
**Current**: `import * as ort from 'onnxruntime-web/webgpu';`
**Problem**: This import path doesn't guarantee WebGPU-enabled runtime. The standalone tests that work use the explicit WebGPU-compiled files.

**Evidence from notes.md**:
> The critical fix was using the **correct ONNX Runtime variant**:
> - ❌ **`ort.min.js`** - Standard ONNX Runtime (no WebGPU support)
> - ✅ **`ort.webgpu.min.js`** - WebGPU-enabled ONNX Runtime

### Issue 2: Cross-Origin Isolation Broken
**Current State**: 
```
crossOriginIsolated: false
hasSharedArrayBuffer: false
```

**Problem**: Without cross-origin isolation:
- SharedArrayBuffer is disabled
- Multi-threaded WASM cannot work (falls back to single-threaded)
- Single-threaded WASM blocks the UI completely
- No way to run async operations properly

**Why It's Broken**:
1. Production nginx proxy strips headers from backend container
2. Frontend nginx sets headers but they get overwritten
3. Headers must be set at the FINAL reverse proxy level

### Issue 3: Module Resolution Confusion
**Current**: Frontend imports `onnxruntime-web/webgpu` via npm package
**Problem**: This doesn't use the WebGPU-compiled WASM files in `/public/onnx/`
**Result**: Falls back to standard WASM, no GPU acceleration

### Issue 4: Overly Complex Async Patterns
The frontend has complex async scheduling logic that doesn't actually prevent blocking:
```typescript
await new Promise(resolve => setTimeout(resolve, 0));
results = await new Promise(async (resolve, reject) => {
  await new Promise(r => setTimeout(r, 10));
  const inferenceResults = await this.session!.run({...}); // STILL BLOCKS
  resolve(inferenceResults);
});
```

**Reality**: setTimeout doesn't help because `session.run()` is synchronous internally. The working tests just call it directly.

## The Solution: 5-Step Fix

### Step 1: Fix Cross-Origin Isolation Headers (CRITICAL)

**File**: `/home/alec/git/pokemon/production_nginx_config.conf`

**Required Changes**:
```nginx
location ^~ /cardcam/ {
    proxy_pass http://localhost:3001/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # CRITICAL: Add cross-origin isolation headers HERE
    add_header Cross-Origin-Embedder-Policy "require-corp" always;
    add_header Cross-Origin-Opener-Policy "same-origin" always;
    add_header Cross-Origin-Resource-Policy "cross-origin" always;
}
```

**Why This Works**:
- Headers are set at the FINAL proxy level (not stripped)
- Enables SharedArrayBuffer
- Enables multi-threaded WASM
- Required for WebGPU to work properly

**Verification**:
```javascript
// In browser console after fix:
console.log(crossOriginIsolated); // Should be true
console.log(typeof SharedArrayBuffer); // Should be "function"
```

### Step 2: Use WebGPU-Specific ONNX Runtime Files

**Current Problem**: npm package import doesn't use the WebGPU-compiled files

**Solution Options**:

#### Option A: Direct Script Loading (Like Working Tests)
```html
<!-- In public/index.html -->
<script src="%PUBLIC_URL%/onnx/ort.webgpu.min.js"></script>
```

Then in TypeScript:
```typescript
// Declare global ort from script tag
declare global {
  interface Window {
    ort: typeof import('onnxruntime-web');
  }
}

// Use it
const ort = window.ort;
```

#### Option B: Use Correct npm Import Path
```typescript
// Instead of generic import
// import * as ort from 'onnxruntime-web/webgpu';

// Use specific WebGPU build
import * as ort from 'onnxruntime-web/dist/ort.webgpu.min.js';
```

**Recommended**: Option A (matches working tests exactly)

### Step 3: Simplify ModelManager to Match Working Tests

**File**: `/home/alec/git/pokemon/frontend/src/utils/ModelManager.ts`

**Remove**:
- Complex async scheduling with setTimeout
- Fallback logic (fails fast instead)
- Multi-threaded detection checks
- Overly verbose logging

**Keep It Simple**:
```typescript
async predict(imageData: ImageData): Promise<ModelPrediction> {
  if (!this.session) {
    throw new Error('Model not loaded');
  }

  const startTime = performance.now();
  
  // Preprocess
  const inputTensor = this.preprocessImage(imageData);
  
  // Run inference - simple and direct
  const results = await this.session.run({
    [this.session.inputNames[0]]: inputTensor
  });
  
  // Postprocess
  const prediction = await this.postprocessResults(results);
  
  console.log(`Inference: ${(performance.now() - startTime).toFixed(2)}ms`);
  return prediction;
}
```

**Why This Works**:
- No setTimeout wrapping (doesn't help anyway)
- Direct session.run() call (like working tests)
- If it blocks, we'll see the real problem (not masked by async patterns)
- With proper cross-origin isolation, WASM won't block

### Step 4: Configure ONNX Runtime Properly

**File**: `/home/alec/git/pokemon/frontend/src/utils/ModelManager.ts`

```typescript
private setupONNXRuntime(): void {
  // WASM paths for fallback
  ort.env.wasm.wasmPaths = (process.env.PUBLIC_URL || '') + '/onnx/';
  
  // Multi-threaded WASM (requires cross-origin isolation)
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
  
  // WebGPU configuration
  if (ort.env.webgpu) {
    ort.env.webgpu.validateInputContent = false;
  }
  
  console.log('ONNX Runtime configured:', {
    crossOriginIsolated,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    threads: ort.env.wasm.numThreads
  });
}
```

**Session Options**:
```typescript
const sessionOptions: ort.InferenceSession.SessionOptions = {
  executionProviders: ['webgpu', 'wasm'],
  graphOptimizationLevel: 'all',
  enableCpuMemArena: true,
  executionMode: 'parallel',
};
```

### Step 5: Verify Execution Provider

**Critical Check**: Verify what's actually being used

```typescript
async loadModel(): Promise<void> {
  this.session = await ort.InferenceSession.create(
    this.config.modelPath,
    sessionOptions
  );
  
  // Verify actual provider
  const handler = (this.session as any).handler;
  const backend = handler?.backend || handler?.backendType;
  
  console.log('✅ Model loaded');
  console.log('📊 Requested providers:', this.config.executionProviders);
  console.log('📊 Actual backend:', backend);
  
  // FAIL FAST if not using WebGPU when requested
  if (this.config.executionProviders[0] === 'webgpu' && backend !== 'webgpu') {
    throw new Error(`Expected WebGPU but got ${backend} - check browser support and headers`);
  }
}
```

## Testing Strategy

### Phase 1: Verify Cross-Origin Isolation
```bash
# 1. Update production nginx config
# 2. Rebuild and restart containers
cd /home/alec/git/pokemon
docker-compose build frontend
docker-compose up -d frontend

# 3. Reload nginx on host
sudo nginx -t
sudo systemctl reload nginx
```

**Verify in browser console**:
```javascript
console.log('Cross-origin isolated:', crossOriginIsolated);
console.log('SharedArrayBuffer:', typeof SharedArrayBuffer);
```

**Expected**: Both should be true/function

### Phase 2: Verify ONNX Runtime
**In browser console**:
```javascript
console.log('window.ort available:', typeof window.ort);
console.log('ONNX Runtime version:', window.ort?.env?.versions);
```

**Expected**: Should show ONNX Runtime is available

### Phase 3: Test Model Loading
**Monitor console for**:
```
✅ Model loaded
📊 Requested providers: ["webgpu", "wasm"]
📊 Actual backend: webgpu
```

**Expected**: Should say "webgpu", not "wasm"

### Phase 4: Test Inference
**Expected behavior**:
- No browser lockup
- Inference completes in <200ms (WebGPU) or <600ms (multi-threaded WASM)
- Console shows actual provider used

## Success Criteria

### WebGPU Working:
- ✅ `crossOriginIsolated === true`
- ✅ Model loads without errors
- ✅ Backend reports `webgpu`
- ✅ Inference ~100-200ms
- ✅ No browser lockups

### WASM Fallback Working:
- ✅ `crossOriginIsolated === true`
- ✅ Multi-threaded WASM enabled
- ✅ Inference ~300-600ms
- ✅ No UI blocking

### Failure Modes to Avoid:
- ❌ `crossOriginIsolated === false` → Single-threaded WASM → UI lockup
- ❌ Backend reports "wasm" when "webgpu" requested → No GPU acceleration
- ❌ Inference >1000ms → Performance problem
- ❌ Browser freezes → Blocking operations

## Why This Will Work

1. **Proven Pattern**: The gpu-acceleration-tests already work with this exact approach
2. **No Guessing**: We know WebGPU works on this hardware (Apple Silicon)
3. **No Fallbacks**: Fail fast exposes real problems instead of masking them
4. **Simple Code**: Less complexity = fewer failure modes
5. **Headers Fixed**: Cross-origin isolation enables all the threading/async features

## What We're NOT Doing

1. ❌ **No Workers**: WebGPU doesn't need workers, session.run() is already async
2. ❌ **No Complex Async**: setTimeout doesn't prevent blocking during session.run()
3. ❌ **No Silent Fallbacks**: If WebGPU fails, we want to know WHY
4. ❌ **No WebNN**: Proven to not work with ONNX Runtime
5. ❌ **No WebGL**: User explicitly rejected this

## Implementation Order

1. **First**: Fix production nginx (enables cross-origin isolation)
2. **Second**: Change ONNX Runtime loading (use WebGPU variant)
3. **Third**: Simplify ModelManager (remove complexity)
4. **Fourth**: Add verification (fail fast on misconfiguration)
5. **Fifth**: Test and validate

## Expected Timeline

- **Step 1 (Headers)**: 5 minutes - Edit nginx config, reload
- **Step 2 (ONNX Runtime)**: 10 minutes - Add script tag, update imports
- **Step 3 (Simplify)**: 15 minutes - Remove complex async patterns
- **Step 4 (Verify)**: 5 minutes - Add logging and checks
- **Step 5 (Test)**: 10 minutes - Rebuild, test, validate

**Total**: ~45 minutes to complete fix

## Final Verification Checklist

After implementation, verify:

```javascript
// Browser console checks
[✓] crossOriginIsolated === true
[✓] typeof SharedArrayBuffer === 'function'
[✓] window.ort !== undefined
[✓] Model loads without errors
[✓] Backend reports 'webgpu' (or 'wasm' with multiple threads)
[✓] Inference completes in <600ms
[✓] No browser lockups
[✓] Camera interface responsive
```

## Rollback Plan

If something goes wrong:
1. **Revert nginx config**: Remove cross-origin headers
2. **Revert ONNX import**: Back to npm package
3. **Keep simple ModelManager**: Even without WebGPU, simpler is better

## Post-Fix Monitoring

Track these metrics:
- Inference time (target: <200ms WebGPU, <600ms WASM)
- Actual execution provider (should match requested)
- Browser lockups (should be zero)
- Cross-origin isolation status (should stay enabled)

---

## The Core Insight

**The problem isn't WebGPU or ONNX Runtime** - the standalone tests prove these work perfectly.

**The problem is:**
1. Cross-origin isolation broken (blocks threading)
2. Wrong ONNX Runtime files loaded (no WebGPU)
3. Over-engineered async patterns (mask real issues)

**The solution is:**
1. Fix the headers
2. Load the right files
3. Keep it simple

**Why we've struggled:**
- Kept adding complexity trying to work around symptoms
- Never fixed the root cause (headers)
- Fallback logic masked the real problems
- Async patterns gave false hope

**Why this will work:**
- We have a working reference (gpu-acceleration-tests)
- We know the hardware supports WebGPU
- We're fixing root causes, not symptoms
- We're keeping it simple and verifiable

