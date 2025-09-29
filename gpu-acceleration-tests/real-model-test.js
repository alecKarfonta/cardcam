// Real Trading Card Detector Model Test with MTG Image

class RealModelTester {
    constructor() {
        this.logs = [];
        this.session = null;
        this.modelConfig = {
            modelPath: './trading_card_detector.onnx',
            inputShape: [1, 3, 1088, 1088], // Your actual model input shape
            outputShape: [1, 300, 7], // [batch, detections, features]
            executionProviders: ['wasm', 'cpu'], // Start with WASM baseline
            confidenceThreshold: 0.25,
            nmsThreshold: 0.45
        };
        this.lastScale = 1.0;
        this.lastPadX = 0;
        this.lastPadY = 0;
        this.lastOrigW = 0;
        this.lastOrigH = 0;
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
        this.logs.push(logEntry);
        console.log(logEntry);
        this.updateLogDisplay();
    }

    updateLogDisplay() {
        const logElement = document.getElementById('realModelLogs');
        if (logElement) {
            logElement.textContent = this.logs.join('\n');
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('realModelStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
    }

    updateMetrics(time, detections) {
        const timeElement = document.getElementById('realModelTime');
        const detectionsElement = document.getElementById('realModelDetections');
        
        if (timeElement) timeElement.textContent = time ? `${time.toFixed(2)}` : '-';
        if (detectionsElement) detectionsElement.textContent = detections !== undefined ? detections : '-';
    }

    setupONNXRuntime(provider = 'wasm') {
        this.log(`Setting up ONNX Runtime for ${provider.toUpperCase()}...`);
        
        // Base configuration
        const base = '/onnx/';
        ort.env.wasm.wasmPaths = base;
        
        if (provider === 'wasm' || provider === 'cpu') {
            // MAXIMUM WASM PERFORMANCE - exactly like your ModelManager
            const maxThreads = Math.max(navigator.hardwareConcurrency || 16, 16);
            ort.env.wasm.numThreads = maxThreads;
            ort.env.wasm.simd = true;
            ort.env.wasm.proxy = false;
            ort.env.wasm.initTimeout = 60000;
            
            this.log(`WASM Configuration:`);
            this.log(`- Threads: ${ort.env.wasm.numThreads} (MAXIMUM CPU utilization)`);
            this.log(`- SIMD: ${ort.env.wasm.simd} (Vectorized operations)`);
            this.log(`- Proxy: ${ort.env.wasm.proxy} (Direct execution)`);
            this.log(`- Cross-origin isolation: ${crossOriginIsolated}`);
            
        } else if (provider === 'webgpu') {
            // WebGPU configuration
            if (ort.env.webgpu) {
                ort.env.webgpu.validateInputContent = false;
                this.log('WebGPU validation disabled for performance');
            }
            
        } else if (provider === 'webnn') {
            // WebNN configuration
            this.log('WebNN configuration - using default settings');
        }
    }

    async loadModel(provider = 'wasm') {
        this.log(`=== Loading Trading Card Detector Model with ${provider.toUpperCase()} ===`);
        this.updateStatus(`Loading model with ${provider}...`, 'info');
        
        try {
            this.setupONNXRuntime(provider);
            
            const loadStartTime = performance.now();
            
            // Check memory before loading
            const memBefore = performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024)
            } : null;
            
            if (memBefore) {
                this.log(`Memory before loading: ${memBefore.used}MB/${memBefore.total}MB`);
            }
            
            // Session options - exactly like your ModelManager
            const sessionOptions = {
                executionProviders: [provider, 'wasm', 'cpu'],
                graphOptimizationLevel: 'all',
                enableMemPattern: true,
                enableCpuMemArena: true,
                executionMode: 'parallel',
                interOpNumThreads: Math.max(navigator.hardwareConcurrency || 16, 16),
                intraOpNumThreads: Math.max(navigator.hardwareConcurrency || 16, 16),
                logSeverityLevel: 3, // Errors only
                logVerbosityLevel: 0
            };
            
            if (provider === 'webgpu') {
                // WebGPU specific options
                sessionOptions.preferredOutputLocation = 'gpu-buffer';
                this.log('WebGPU: preferredOutputLocation set to gpu-buffer');
            }
            
            this.log(`Session options: ${JSON.stringify(sessionOptions, null, 2)}`);
            
            // Create session
            const sessionStart = performance.now();
            this.log('Creating ONNX Runtime session...');
            
            try {
                this.session = await ort.InferenceSession.create(
                    this.modelConfig.modelPath,
                    sessionOptions
                );
                this.log('✅ Session created successfully');
                
            } catch (sessionError) {
                this.log(`❌ Session creation failed: ${sessionError.message}`, 'error');
                
                if (provider !== 'wasm') {
                    this.log('🔄 Falling back to WASM...', 'warning');
                    const fallbackOptions = {
                        ...sessionOptions,
                        executionProviders: ['wasm', 'cpu'],
                        preferredOutputLocation: undefined
                    };
                    
                    this.session = await ort.InferenceSession.create(
                        this.modelConfig.modelPath,
                        fallbackOptions
                    );
                    this.log('✅ WASM fallback session created');
                } else {
                    throw sessionError;
                }
            }
            
            const sessionTime = performance.now() - sessionStart;
            const totalLoadTime = performance.now() - loadStartTime;
            
            // Check memory after loading
            const memAfter = performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024)
            } : null;
            
            this.log('✅ MODEL LOADING COMPLETED!');
            this.log(`Input names: ${this.session.inputNames.join(', ')}`);
            this.log(`Output names: ${this.session.outputNames.join(', ')}`);
            this.log(`Session creation: ${sessionTime.toFixed(2)}ms`);
            this.log(`Total load time: ${totalLoadTime.toFixed(2)}ms`);
            
            if (memBefore && memAfter) {
                this.log(`Memory after loading: ${memAfter.used}MB/${memAfter.total}MB (Δ${memAfter.used - memBefore.used}MB)`);
            }
            
            // Check actual execution provider
            const sessionAny = this.session;
            this.log(`Actual execution provider: ${sessionAny.executionProviders || 'unknown'}`);
            
            this.updateStatus(`Model loaded successfully with ${provider}`, 'success');
            return true;
            
        } catch (error) {
            this.log(`❌ Model loading failed: ${error.message}`, 'error');
            this.updateStatus(`Model loading failed: ${error.message}`, 'error');
            return false;
        }
    }

    async loadTestImage() {
        this.log('Loading MTG test image...');
        
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = () => {
                try {
                    // Create canvas and get image data
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    
                    const imageData = ctx.getImageData(0, 0, img.width, img.height);
                    this.log(`✅ Image loaded: ${img.width}x${img.height} pixels`);
                    resolve(imageData);
                    
                } catch (error) {
                    reject(error);
                }
            };
            
            img.onerror = () => {
                reject(new Error('Failed to load MTG test image'));
            };
            
            img.src = './mtg.png';
        });
    }

    preprocessImage(imageData) {
        const { width, height } = imageData;
        const [, , targetH, targetW] = this.modelConfig.inputShape; // 1088x1088

        this.log(`Preprocessing image: ${width}x${height} → ${targetW}x${targetH}`);

        // Calculate letterbox parameters - exactly like your ModelManager
        const r = Math.min(targetW / width, targetH / height);
        const newW = Math.round(width * r);
        const newH = Math.round(height * r);
        const padX = Math.floor((targetW - newW) / 2);
        const padY = Math.floor((targetH - newH) / 2);
        
        // Store for postprocessing
        this.lastScale = r;
        this.lastPadX = padX;
        this.lastPadY = padY;
        this.lastOrigW = width;
        this.lastOrigH = height;
        
        this.log(`Letterbox: scale=${r.toFixed(3)}, pad=${padX}x${padY}, new_size=${newW}x${newH}`);

        // Canvas operations
        const srcCanvas = document.createElement('canvas');
        const srcCtx = srcCanvas.getContext('2d', { alpha: false });
        srcCanvas.width = width;
        srcCanvas.height = height;
        srcCtx.putImageData(imageData, 0, 0);

        const dstCanvas = document.createElement('canvas');
        const dstCtx = dstCanvas.getContext('2d', { 
            alpha: false,
            willReadFrequently: true
        });
        dstCanvas.width = targetW;
        dstCanvas.height = targetH;
        
        // Fill with gray background
        dstCtx.fillStyle = '#808080';
        dstCtx.fillRect(0, 0, targetW, targetH);
        
        // Draw resized image
        dstCtx.imageSmoothingEnabled = false;
        dstCtx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newW, newH);

        // Get resized image data
        const resized = dstCtx.getImageData(0, 0, targetW, targetH);
        const pixels = resized.data;
        
        // Create tensor - exactly like your ModelManager
        const tensorData = new Float32Array(1 * 3 * targetH * targetW);
        const pixelCount = targetH * targetW;
        
        const inv255 = 1.0 / 255.0;
        for (let i = 0; i < pixelCount; i++) {
            const p = i * 4;
            tensorData[i] = pixels[p] * inv255;                    // R
            tensorData[i + pixelCount] = pixels[p + 1] * inv255;   // G  
            tensorData[i + 2 * pixelCount] = pixels[p + 2] * inv255; // B
        }
        
        const tensor = new ort.Tensor('float32', tensorData, this.modelConfig.inputShape);
        this.log(`✅ Tensor created: ${tensor.dims} (${tensorData.length} elements)`);
        
        return tensor;
    }

    async runInference(provider = 'wasm') {
        this.log(`=== Running Inference with ${provider.toUpperCase()} ===`);
        this.updateStatus(`Running inference with ${provider}...`, 'info');
        
        try {
            if (!this.session) {
                const loaded = await this.loadModel(provider);
                if (!loaded) {
                    throw new Error('Failed to load model');
                }
            }
            
            // Load test image
            const imageData = await this.loadTestImage();
            
            const overallStart = performance.now();
            
            // Preprocessing
            this.log('🔍 PREPROCESSING: Starting...');
            const preprocessStart = performance.now();
            const inputTensor = this.preprocessImage(imageData);
            const preprocessTime = performance.now() - preprocessStart;
            this.log(`✅ PREPROCESSING: ${preprocessTime.toFixed(2)}ms`);
            
            // Inference
            this.log(`🔍 INFERENCE: Starting with ${provider.toUpperCase()}...`);
            const inferenceStart = performance.now();
            
            // Check memory before inference
            const memBefore = performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024)
            } : null;
            
            this.log(`About to call session.run() - this is where WebGPU typically locks up...`);
            
            let results;
            if (provider === 'webgpu') {
                // For WebGPU, we'll add a timeout to detect lockups
                results = await Promise.race([
                    this.session.run({
                        [this.session.inputNames[0]]: inputTensor
                    }),
                    new Promise((_, reject) => {
                        setTimeout(() => {
                            reject(new Error('WebGPU inference timeout - likely browser lockup'));
                        }, 30000); // 30 second timeout
                    })
                ]);
            } else {
                // Standard inference for other providers
                results = await this.session.run({
                    [this.session.inputNames[0]]: inputTensor
                });
            }
            
            const inferenceTime = performance.now() - inferenceStart;
            this.log(`✅ INFERENCE: ${inferenceTime.toFixed(2)}ms`);
            
            // Check memory after inference
            const memAfter = performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024)
            } : null;
            
            if (memBefore && memAfter) {
                this.log(`Memory: ${memBefore.used}MB → ${memAfter.used}MB (Δ${memAfter.used - memBefore.used}MB)`);
            }
            
            // Postprocessing
            this.log('🔍 POSTPROCESSING: Starting...');
            const postprocessStart = performance.now();
            const detections = await this.postprocessResults(results);
            const postprocessTime = performance.now() - postprocessStart;
            this.log(`✅ POSTPROCESSING: ${postprocessTime.toFixed(2)}ms`);
            
            const totalTime = performance.now() - overallStart;
            
            this.log(`🎯 PERFORMANCE SUMMARY (${provider.toUpperCase()}):`);
            this.log(`   Preprocessing: ${preprocessTime.toFixed(2)}ms (${(preprocessTime/totalTime*100).toFixed(1)}%)`);
            this.log(`   Inference: ${inferenceTime.toFixed(2)}ms (${(inferenceTime/totalTime*100).toFixed(1)}%)`);
            this.log(`   Postprocessing: ${postprocessTime.toFixed(2)}ms (${(postprocessTime/totalTime*100).toFixed(1)}%)`);
            this.log(`   Total: ${totalTime.toFixed(2)}ms`);
            
            this.log(`🎯 DETECTIONS: Found ${detections.validDetections} trading cards`);
            
            this.updateMetrics(totalTime, detections.validDetections);
            this.updateStatus(`Inference completed! Found ${detections.validDetections} detections`, 'success');
            
            // Display detection details
            if (detections.validDetections > 0) {
                this.log('Detection details:');
                for (let i = 0; i < detections.validDetections; i++) {
                    const score = detections.scores[i];
                    const classId = detections.classes[i];
                    this.log(`  Detection ${i + 1}: confidence=${score.toFixed(3)}, class=${classId}`);
                }
            }
            
            return detections;
            
        } catch (error) {
            this.log(`❌ INFERENCE FAILED: ${error.message}`, 'error');
            this.updateStatus(`Inference failed: ${error.message}`, 'error');
            
            if (error.message.includes('timeout') || error.message.includes('lockup')) {
                this.log('🚨 BROWSER LOCKUP DETECTED - This confirms WebGPU blocking behavior', 'error');
            }
            
            throw error;
        }
    }

    async postprocessResults(results) {
        const outputNames = Object.keys(results);
        const output = results[outputNames[0]];
        
        if (!output) {
            throw new Error('Invalid model output');
        }
        
        // Handle GPU buffer downloads
        let outputData;
        if (output.location === 'gpu-buffer') {
            this.log('📥 Downloading GPU data to CPU...');
            outputData = await output.getData();
        } else {
            outputData = output.data;
        }
        
        const dims = output.dims;
        this.log(`Output shape: ${dims}`);
        this.log(`Output data length: ${outputData.length}`);
        this.log(`First 10 values: ${Array.from(outputData.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}`);
        
        // Process detections - format: [1, 300, 7] where each detection is [cx, cy, w, h, conf, class, angle]
        if (!dims || dims.length !== 3 || dims[2] !== 7) {
            throw new Error(`Unexpected output format. Expected [1, 300, 7], got ${dims}`);
        }
        
        const [, maxDetections, featuresPerDetection] = dims;
        const boxes = [];
        const scores = [];
        const classes = [];
        const rotatedBoxes = [];
        
        this.log(`Processing ${maxDetections} potential detections...`);
        
        for (let i = 0; i < maxDetections; i++) {
            const detectionOffset = i * featuresPerDetection;
            const detection = outputData.slice(detectionOffset, detectionOffset + featuresPerDetection);
            
            // Skip empty detections
            if (detection.every(val => val === 0)) continue;
            
            const cx = detection[0];
            const cy = detection[1];
            const w = detection[2];
            const h = detection[3];
            const confidence = detection[4];
            const classId = detection[5];
            const angle = detection[6];
            
            // Filter by confidence
            if (confidence <= this.modelConfig.confidenceThreshold) continue;
            
            this.log(`Detection ${i}: cx=${cx.toFixed(1)}, cy=${cy.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}, conf=${confidence.toFixed(3)}, angle=${angle.toFixed(3)}`);
            
            // Convert to normalized coordinates - exactly like your ModelManager
            const halfW = w / 2;
            const halfH = h / 2;
            const cosT = Math.cos(angle);
            const sinT = Math.sin(angle);
            
            const corners = [
                { x: cx + (-halfW * cosT - (-halfH) * sinT), y: cy + (-halfW * sinT + (-halfH) * cosT) },
                { x: cx + (halfW * cosT - (-halfH) * sinT), y: cy + (halfW * sinT + (-halfH) * cosT) },
                { x: cx + (halfW * cosT - halfH * sinT), y: cy + (halfW * sinT + halfH * cosT) },
                { x: cx + (-halfW * cosT - halfH * sinT), y: cy + (-halfW * sinT + halfH * cosT) }
            ];
            
            // Map back from letterbox to original image coordinates
            const originalCorners = corners.map(corner => ({
                x: (corner.x - this.lastPadX) / this.lastScale,
                y: (corner.y - this.lastPadY) / this.lastScale,
            }));
            
            // Normalize to [0, 1] range
            const normalizedCorners = originalCorners.map(corner => ({
                x: Math.max(0, Math.min(1, corner.x / this.lastOrigW)),
                y: Math.max(0, Math.min(1, corner.y / this.lastOrigH)),
            }));
            
            // Calculate axis-aligned bounding box
            const xs = normalizedCorners.map(p => p.x);
            const ys = normalizedCorners.map(p => p.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            const widthN = maxX - minX;
            const heightN = maxY - minY;
            
            // Filter tiny boxes
            const minArea = 0.001;
            if (widthN * heightN < minArea) continue;
            if (minX >= 1 || maxX <= 0 || minY >= 1 || maxY <= 0) continue;
            
            // Store results
            boxes.push(minX, minY, widthN, heightN);
            scores.push(confidence);
            classes.push(Math.round(classId));
            
            // Store rotated corners
            rotatedBoxes.push(
                normalizedCorners[0].x, normalizedCorners[0].y,
                normalizedCorners[1].x, normalizedCorners[1].y,
                normalizedCorners[2].x, normalizedCorners[2].y,
                normalizedCorners[3].x, normalizedCorners[3].y,
            );
        }
        
        this.log(`Final results: ${boxes.length / 4} valid detections`);
        
        return {
            boxes: new Float32Array(boxes),
            rotatedBoxes: rotatedBoxes.length > 0 ? new Float32Array(rotatedBoxes) : undefined,
            scores: new Float32Array(scores),
            classes: new Int32Array(classes),
            validDetections: boxes.length / 4
        };
    }

    clearLogs() {
        this.logs = [];
        this.updateLogDisplay();
        this.updateStatus('Logs cleared', 'info');
        this.updateMetrics(null, null);
    }

    async dispose() {
        if (this.session) {
            await this.session.release();
            this.session = null;
        }
    }
}

// Global real model tester instance
let realModelTester;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    realModelTester = new RealModelTester();
});

// Global functions for HTML buttons
function testRealModelWASM() {
    if (realModelTester) {
        realModelTester.runInference('wasm');
    }
}

function testRealModelWebGPU() {
    if (realModelTester) {
        realModelTester.runInference('webgpu');
    }
}

function testRealModelWebNN() {
    if (realModelTester) {
        realModelTester.runInference('webnn');
    }
}

function clearRealModelLogs() {
    if (realModelTester) {
        realModelTester.clearLogs();
    }
}

// Export for use in other modules
window.RealModelTester = RealModelTester;
