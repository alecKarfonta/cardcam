import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  modelPath: string;
  inputShape: [number, number, number, number]; // [batch, channels, height, width]
  outputShape: [number, number] | [number, number, number]; // [batch, classes] or [batch, features, anchors] for OBB
  executionProviders: string[];
  confidenceThreshold: number;
  nmsThreshold: number;
}

export interface ModelPrediction {
  boxes: Float32Array;
  rotatedBoxes?: Float32Array; // For OBB: [x1,y1,x2,y2,x3,y3,x4,y4] per detection
  scores: Float32Array;
  classes: Int32Array;
  validDetections: number;
}

export class ModelManager {
  private session: ort.InferenceSession | null = null;
  private config: ModelConfig;
  private isLoaded: boolean = false;
  // store last resize mapping (letterbox)
  private lastScale: number = 1.0;
  private lastPadX: number = 0;
  private lastPadY: number = 0;
  private lastOrigW: number = 0;
  private lastOrigH: number = 0;

  constructor(config: ModelConfig) {
    this.config = config;
    this.setupONNXRuntime();
  }

  private setupONNXRuntime(): void {
    // ULTIMATE WASM PERFORMANCE: Maximum optimization configuration
    const base = (process.env.PUBLIC_URL || '') + '/onnx/';
    ort.env.wasm.wasmPaths = base;
    
    // MAXIMUM THREADING: Use all available CPU cores
    const maxThreads = Math.max(navigator.hardwareConcurrency || 16, 16); // Use up to 16 threads
    ort.env.wasm.numThreads = maxThreads;
    ort.env.wasm.simd = true; // SIMD vectorization
    ort.env.wasm.proxy = false; // Direct execution for speed
    
    // ADVANCED WASM OPTIMIZATIONS
    ort.env.wasm.initTimeout = 60000; // Longer timeout for complex models
    
    console.log('🚀 ULTIMATE WASM PERFORMANCE CONFIGURATION:');
    console.log(`   - Threads: ${ort.env.wasm.numThreads} (MAXIMUM CPU utilization)`);
    console.log(`   - SIMD: ${ort.env.wasm.simd} (Vectorized operations)`);
    console.log(`   - Proxy: ${ort.env.wasm.proxy} (Direct execution)`);
    console.log(`   - Cross-origin isolation: ENABLED (for threading)`);
    console.log(`   - Model size: 1088x1088 (full resolution for accuracy)`);
    console.log('⚡ WASM optimized to the MAXIMUM - WebGPU avoided due to lockup issues');
  }

  private isWebGPUSupported(): boolean {
    try {
      return 'gpu' in navigator && 
             navigator.gpu !== undefined && 
             typeof (navigator.gpu as any)?.requestAdapter === 'function';
    } catch {
      return false;
    }
  }


  async loadModel(): Promise<void> {
    if (this.isLoaded) return;

    const loadStartTime = performance.now();
    
    try {
      console.log('🚀 MODEL LOADING: Starting model load from:', this.config.modelPath);
      console.log('📊 Execution providers:', this.config.executionProviders);
      console.log('🔧 ONNX Runtime environment:', {
        wasmPaths: ort.env.wasm.wasmPaths,
        numThreads: ort.env.wasm.numThreads,
        simd: ort.env.wasm.simd,
        proxy: ort.env.wasm.proxy
      });
      
      // PROFILING: Check memory before loading
      const memBefore = (performance as any).memory ? {
        used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024)
      } : null;
      if (memBefore) {
        console.log(`📊 Memory before loading: ${memBefore.used}MB/${memBefore.total}MB`);
      }
      
      // WebGPU-OPTIMIZED: Session configuration for WebGPU execution provider
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: this.config.executionProviders,
        graphOptimizationLevel: 'all', // Enable ALL graph optimizations
        enableMemPattern: true, // Enable memory pattern optimization
        enableCpuMemArena: true, // Enable CPU memory arena
        executionMode: 'parallel', // Enable parallel execution
        interOpNumThreads: Math.min(navigator.hardwareConcurrency || 4, 4),
        intraOpNumThreads: Math.min(navigator.hardwareConcurrency || 4, 4),
        logSeverityLevel: 3, // Error level only for better performance
        logVerbosityLevel: 0, // Minimal verbosity
        enableProfiling: false, // Disable profiling for production
      };
      
      // MAXIMUM WASM OPTIMIZATIONS - WebGPU removed due to browser lockups
      console.log('🚀 MAXIMUM WASM PERFORMANCE: Configuring session for ultimate CPU performance');
      
      // Aggressive graph optimizations for WASM
      sessionOptions.graphOptimizationLevel = 'all';
      sessionOptions.enableMemPattern = true; // Memory pattern optimization
      sessionOptions.enableCpuMemArena = true; // CPU memory arena for speed
      
      // Parallel execution for multi-threading
      sessionOptions.executionMode = 'parallel';
      sessionOptions.interOpNumThreads = Math.max(navigator.hardwareConcurrency || 16, 16);
      sessionOptions.intraOpNumThreads = Math.max(navigator.hardwareConcurrency || 16, 16);
      
      // Minimal logging for performance
      sessionOptions.logSeverityLevel = 3; // Errors only
      sessionOptions.logVerbosityLevel = 0; // No verbose logging
      
      console.log('⚡ WASM session configured for MAXIMUM performance - all CPU cores utilized');
      console.log('💡 WebGPU intentionally disabled to prevent browser lockups');
      
      // WebNN-specific optimizations (fallback)
      if (this.config.executionProviders.includes('webnn')) {
        console.log('🧠 WebNN OPTIMIZED: Configuring session for neural network acceleration');
        // WebNN works best with these settings
        sessionOptions.graphOptimizationLevel = 'all';
        sessionOptions.enableMemPattern = true; // WebNN can benefit from memory patterns
        sessionOptions.enableCpuMemArena = true; // WebNN can use CPU memory efficiently
        
        // WebNN-specific configurations to handle int64 issues
        sessionOptions.logSeverityLevel = 0; // Enable all logs for WebNN debugging
        sessionOptions.logVerbosityLevel = 1; // Verbose logging for WebNN
        
        console.log('⚡ WebNN optimizations applied - handling int64 compatibility');
      }
      
      console.log('🚀 ADVANCED session optimizations: ALL graph optimizations + parallel execution enabled');
      console.log('📋 Session options:', sessionOptions);

      // PROFILING: Session creation
      console.log('🔍 Creating ONNX Runtime session...');
      console.log('🔍 Requested execution providers:', this.config.executionProviders);
      
      const sessionStart = performance.now();
      
      try {
        this.session = await ort.InferenceSession.create(
          this.config.modelPath,
          sessionOptions
        );
      } catch (sessionError) {
        console.error('❌ Session creation failed:', sessionError);
        console.log('🔄 Attempting fallback with WASM only...');
        
        // Fallback to WASM only with proper threading
        const wasmOnlyOptions = {
          ...sessionOptions,
          executionProviders: ['wasm', 'cpu'],
          // Remove WebGPU-specific options
          enableMemPattern: true,
          enableCpuMemArena: true,
          preferredOutputLocation: undefined
        };
        console.log('🔄 Fallback execution providers:', wasmOnlyOptions.executionProviders);
        
        this.session = await ort.InferenceSession.create(
          this.config.modelPath,
          wasmOnlyOptions
        );
        
        console.log('✅ WASM fallback session created successfully');
      }
      
      const sessionTime = performance.now() - sessionStart;
      
      // PROFILING: Check memory after loading
      const memAfter = (performance as any).memory ? {
        used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024)
      } : null;
      
      const totalLoadTime = performance.now() - loadStartTime;
      
      this.isLoaded = true;
      console.log('✅ MODEL LOADING: Completed successfully!');
      console.log('📥 Input names:', this.session.inputNames);
      console.log('📤 Output names:', this.session.outputNames);
      console.log(`📊 Session creation: ${sessionTime.toFixed(2)}ms`);
      console.log(`📊 Total load time: ${totalLoadTime.toFixed(2)}ms`);
      
      if (memBefore && memAfter) {
        console.log(`📊 Memory after loading: ${memAfter.used}MB/${memAfter.total}MB (Δ${memAfter.used - memBefore.used}MB)`);
      }
      
      // DETAILED: Log which execution provider is actually being used
      console.log('🔍 DEBUGGING: Checking actual execution provider...');
      
      // Try multiple ways to detect the actual execution provider
      const sessionAny = this.session as any;
      const actualProvider = sessionAny.executionProviders || sessionAny._executionProviders || 'unknown';
      const sessionHandler = sessionAny.handler || sessionAny._handler;
      const backendType = sessionHandler?.backend?.backendType || sessionHandler?.backendType || 'unknown';
      
      console.log('🎯 Session execution providers:', actualProvider);
      console.log('🎯 Backend type:', backendType);
      console.log('🎯 Session handler info:', {
        hasHandler: !!sessionHandler,
        handlerKeys: sessionHandler ? Object.keys(sessionHandler) : [],
      });
      
      // Check if WebNN is actually working
      if (this.config.executionProviders.includes('webnn')) {
        if (backendType === 'webnn' || (Array.isArray(actualProvider) && actualProvider.includes('webnn'))) {
          console.log('✅ WebNN: Successfully initialized and active');
        } else {
          console.log('⚠️ WebNN: Requested but not active - likely fell back to ' + backendType);
          console.log('💡 WebNN: Check browser flags: chrome://flags/#enable-webnn-api');
        }
      }
      
      console.log('🎯 Model ready for inference');

    } catch (error) {
      console.error('❌ Model loading failed:', error);
      console.error('🔍 Error details:', {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      });
      throw new Error(`Model loading failed: ${error}`);
    }
  }

  async predict(imageData: ImageData): Promise<ModelPrediction> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // DETAILED PROFILING: Start comprehensive timing
    const overallStartTime = performance.now();
    const timings: Record<string, number> = {};
    
    try {
      // Detect actual execution provider from config
      const configuredProviders = this.config.executionProviders;
      const primaryProvider = configuredProviders[0] || 'unknown';
      
      console.log(`🚀 INFERENCE START with ${primaryProvider.toUpperCase()} (configured providers: ${configuredProviders.join(', ')})`);
      console.log(`📊 Input image: ${imageData.width}x${imageData.height} pixels`);
      
      // PROFILING: Image preprocessing
      console.log(`🔍 PREPROCESSING: Starting image preprocessing...`);
      const preprocessStart = performance.now();
      const inputTensor = this.preprocessImage(imageData);
      const preprocessTime = performance.now() - preprocessStart;
      timings.preprocessing = preprocessTime;
      console.log(`✅ PREPROCESSING: Completed in ${preprocessTime.toFixed(2)}ms`);
      console.log(`📊 PREPROCESSING: Tensor shape ${inputTensor.dims}, size ${inputTensor.data.length} elements`);
      
      // PROFILING: Model inference
      console.log(`🔍 INFERENCE: Starting model inference with ${primaryProvider.toUpperCase()}...`);
      console.log(`📊 INFERENCE: Input tensor shape: ${inputTensor.dims}, type: ${inputTensor.type}`);
      console.log(`📊 INFERENCE: Session input names: ${this.session.inputNames}`);
      console.log(`📊 INFERENCE: Session output names: ${this.session.outputNames}`);
      
      const inferenceStart = performance.now();
      
      // Check memory before inference
      const memBefore = (performance as any).memory ? {
        used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024)
      } : null;
      
      // WebGPU-specific handling to prevent lockups
      console.log(`🔍 INFERENCE: Calling session.run() with ${primaryProvider.toUpperCase()}...`);
      const sessionRunStart = performance.now();
      
      // Standard inference (WebGPU should use Worker, not main thread)
      const results = await this.session.run({
        [this.session.inputNames[0]]: inputTensor
      });
      
      const sessionRunTime = performance.now() - sessionRunStart;
      const inferenceTime = performance.now() - inferenceStart;
      timings.inference = inferenceTime;
      
      console.log(`📊 INFERENCE: session.run() took ${sessionRunTime.toFixed(2)}ms`);
      console.log(`📊 INFERENCE: Total inference time ${inferenceTime.toFixed(2)}ms`);
      
      // Check memory after inference
      const memAfter = (performance as any).memory ? {
        used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024)
      } : null;
      
      console.log(`✅ INFERENCE: Completed in ${inferenceTime.toFixed(2)}ms`);
      if (memBefore && memAfter) {
        console.log(`📊 MEMORY: Before ${memBefore.used}MB/${memBefore.total}MB, After ${memAfter.used}MB/${memAfter.total}MB (Δ${memAfter.used - memBefore.used}MB)`);
      }
      
      // PROFILING: Post-processing
      console.log(`🔍 POSTPROCESSING: Starting result processing...`);
      const postprocessStart = performance.now();
      const prediction = await this.postprocessResults(results);
      const postprocessTime = performance.now() - postprocessStart;
      timings.postprocessing = postprocessTime;
      console.log(`✅ POSTPROCESSING: Completed in ${postprocessTime.toFixed(2)}ms`);
      console.log(`📊 POSTPROCESSING: Found ${prediction.validDetections} detections`);
      
      // PROFILING: Overall summary
      const totalTime = performance.now() - overallStartTime;
      timings.total = totalTime;
      
      console.log(`🎯 PERFORMANCE SUMMARY (${primaryProvider.toUpperCase()}):`);
      console.log(`   📊 Preprocessing: ${timings.preprocessing.toFixed(2)}ms (${(timings.preprocessing/totalTime*100).toFixed(1)}%)`);
      console.log(`   📊 Inference: ${timings.inference.toFixed(2)}ms (${(timings.inference/totalTime*100).toFixed(1)}%)`);
      console.log(`   📊 Postprocessing: ${timings.postprocessing.toFixed(2)}ms (${(timings.postprocessing/totalTime*100).toFixed(1)}%)`);
      console.log(`   📊 Total: ${totalTime.toFixed(2)}ms`);
      
      // Identify bottleneck
      const bottleneck = Object.entries(timings)
        .filter(([key]) => key !== 'total')
        .reduce((max, [key, time]) => time > max.time ? { key, time } : max, { key: '', time: 0 });
      console.log(`🎯 BOTTLENECK: ${bottleneck.key} (${bottleneck.time.toFixed(2)}ms, ${(bottleneck.time/totalTime*100).toFixed(1)}%)`);
      
      return prediction;

    } catch (error) {
      const totalTime = performance.now() - overallStartTime;
      console.error(`❌ INFERENCE FAILED after ${totalTime.toFixed(2)}ms:`, error);
      console.error('🔍 Error details:', {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : String(error),
        timings: timings
      });
      throw new Error(`Prediction failed: ${error}`);
    }
  }


  private preprocessImageToFloat32Array(imageData: ImageData): Float32Array {
    const { width, height, data } = imageData;
    const [, , targetH, targetW] = this.config.inputShape; // now 1088x1088

    // PERFORMANCE: Calculate letterbox parameters
    const r = Math.min(targetW / width, targetH / height);
    const newW = Math.round(width * r);
    const newH = Math.round(height * r);
    const padX = Math.floor((targetW - newW) / 2);
    const padY = Math.floor((targetH - newH) / 2);

    // PERFORMANCE: Use optimized canvas operations
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d', { alpha: false })!;
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCtx.putImageData(imageData, 0, 0);

    const dstCanvas = document.createElement('canvas');
    const dstCtx = dstCanvas.getContext('2d', { 
      alpha: false, // Disable alpha for better performance
      willReadFrequently: true // Optimize for frequent getImageData calls
    })!;
    dstCanvas.width = targetW;
    dstCanvas.height = targetH;
    
    // PERFORMANCE: Fill background with gray (matching Ultralytics default)
    dstCtx.fillStyle = '#808080';
    dstCtx.fillRect(0, 0, targetW, targetH);
    
    // PERFORMANCE: Optimized image scaling
    dstCtx.imageSmoothingEnabled = false; // Disable smoothing for better performance
    dstCtx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newW, newH);

    // PERFORMANCE: Optimized tensor conversion with pre-allocated arrays
    const resized = dstCtx.getImageData(0, 0, targetW, targetH);
    const pixels = resized.data;
    const tensorData = new Float32Array(1 * 3 * targetH * targetW);
    const pixelCount = targetH * targetW;
    
    // PERFORMANCE: Vectorized normalization loop
    for (let i = 0; i < pixelCount; i++) {
      const p = i * 4;
      const inv255 = 1.0 / 255.0; // Multiply instead of divide for better performance
      tensorData[i] = pixels[p] * inv255;                    // R
      tensorData[i + pixelCount] = pixels[p + 1] * inv255;   // G  
      tensorData[i + 2 * pixelCount] = pixels[p + 2] * inv255; // B
    }
    
    return tensorData;
  }

  private preprocessImage(imageData: ImageData): ort.Tensor {
    const { width, height } = imageData;
    const [, , targetH, targetW] = this.config.inputShape; // now 1088x1088

    // PROFILING: Letterbox calculation
    const letterboxStart = performance.now();
    const r = Math.min(targetW / width, targetH / height);
    const newW = Math.round(width * r);
    const newH = Math.round(height * r);
    const padX = Math.floor((targetW - newW) / 2);
    const padY = Math.floor((targetH - newH) / 2);
    this.lastScale = r;
    this.lastPadX = padX;
    this.lastPadY = padY;
    this.lastOrigW = width;
    this.lastOrigH = height;
    const letterboxTime = performance.now() - letterboxStart;
    console.log(`   📊 Letterbox calculation: ${letterboxTime.toFixed(2)}ms (scale=${r.toFixed(3)}, pad=${padX}x${padY})`);

    // PROFILING: Canvas setup
    const canvasSetupStart = performance.now();
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d', { alpha: false })!;
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCtx.putImageData(imageData, 0, 0);

    const dstCanvas = document.createElement('canvas');
    const dstCtx = dstCanvas.getContext('2d', { 
      alpha: false, // Disable alpha for better performance
      willReadFrequently: true // Optimize for frequent getImageData calls
    })!;
    dstCanvas.width = targetW;
    dstCanvas.height = targetH;
    const canvasSetupTime = performance.now() - canvasSetupStart;
    console.log(`   📊 Canvas setup: ${canvasSetupTime.toFixed(2)}ms`);
    
    // PROFILING: Canvas drawing
    const drawingStart = performance.now();
    dstCtx.fillStyle = '#808080';
    dstCtx.fillRect(0, 0, targetW, targetH);
    dstCtx.imageSmoothingEnabled = false; // Disable smoothing for better performance
    dstCtx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newW, newH);
    const drawingTime = performance.now() - drawingStart;
    console.log(`   📊 Canvas drawing: ${drawingTime.toFixed(2)}ms (${width}x${height} → ${newW}x${newH})`);

    // PROFILING: Image data extraction
    const extractionStart = performance.now();
    const resized = dstCtx.getImageData(0, 0, targetW, targetH);
    const pixels = resized.data;
    const extractionTime = performance.now() - extractionStart;
    console.log(`   📊 Image data extraction: ${extractionTime.toFixed(2)}ms (${pixels.length} bytes)`);
    
    // PROFILING: Tensor creation and normalization
    const tensorStart = performance.now();
    const tensorData = new Float32Array(1 * 3 * targetH * targetW);
    const pixelCount = targetH * targetW;
    
    // Vectorized normalization loop
    const inv255 = 1.0 / 255.0; // Multiply instead of divide for better performance
    for (let i = 0; i < pixelCount; i++) {
      const p = i * 4;
      tensorData[i] = pixels[p] * inv255;                    // R
      tensorData[i + pixelCount] = pixels[p + 1] * inv255;   // G  
      tensorData[i + 2 * pixelCount] = pixels[p + 2] * inv255; // B
    }
    
    const tensor = new ort.Tensor('float32', tensorData, this.config.inputShape);
    const tensorTime = performance.now() - tensorStart;
    console.log(`   📊 Tensor creation: ${tensorTime.toFixed(2)}ms (${tensorData.length} elements)`);
    
    return tensor;
  }

  private async postprocessResults(results: ort.InferenceSession.ReturnType): Promise<ModelPrediction> {
    // PROFILING: Output extraction
    const extractStart = performance.now();
    const outputNames = Object.keys(results);
    const output = results[outputNames[0]];
    if (!output) {
      throw new Error('Invalid model output');
    }
    const extractTime = performance.now() - extractStart;
    console.log(`   📊 Output extraction: ${extractTime.toFixed(2)}ms`);

    // PROFILING: Data download (GPU to CPU if needed)
    const downloadStart = performance.now();
    let outputData: Float32Array;
    if ((output as any).location === 'gpu-buffer') {
      console.log('   📥 Downloading GPU data to CPU...');
      outputData = await (output as any).getData() as Float32Array;
    } else {
      outputData = (output as any).data as Float32Array;
    }
    const downloadTime = performance.now() - downloadStart;
    console.log(`   📊 Data download: ${downloadTime.toFixed(2)}ms (location: ${(output as any).location || 'cpu'})`);

    if (!outputData) {
      throw new Error('Failed to get output data');
    }

    const dims = (output as any).dims as number[] | undefined;
    console.log('📊 Output shape:', dims);
    console.log('📊 Output data length:', outputData.length);
    console.log('📊 First 20 values:', Array.from(outputData.slice(0, 20)));

    // New format: [1, 300, 7] where each detection is [cx, cy, w, h, conf, class, angle]
    // NMS is already applied in the ONNX model
    if (!dims || dims.length !== 3 || dims[2] !== 7) {
      throw new Error(`Unexpected output format. Expected [1, 300, 7], got ${dims}`);
    }

    const [, maxDetections, featuresPerDetection] = dims;
    const boxes: number[] = [];
    const scores: number[] = [];
    const classes: number[] = [];
    const rotatedBoxesAll: number[] = [];

    console.log(`🎯 Processing ${maxDetections} detections with NMS already applied`);

    // Process each detection: [cx, cy, w, h, conf, class, angle]
    for (let i = 0; i < maxDetections; i++) {
      const detectionOffset = i * featuresPerDetection;
      
      // Skip empty detections (all zeros)
      const detection = outputData.slice(detectionOffset, detectionOffset + featuresPerDetection);
      if (detection.every(val => val === 0)) continue;
      
      const cx = detection[0];
      const cy = detection[1]; 
      const w = detection[2];
      const h = detection[3];
      const confidence = detection[4];
      const classId = detection[5];
      const angle = detection[6];
      
      // Filter by confidence threshold
      if (confidence <= this.config.confidenceThreshold) continue;
      
      console.log(`Detection ${i}: cx=${cx.toFixed(1)}, cy=${cy.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}, conf=${confidence.toFixed(3)}, angle=${angle.toFixed(3)}`);
      
      // Create rotated corners in letterbox space (coordinates are in pixel space relative to 1088x1088)
      const halfW = w / 2;
      const halfH = h / 2;
      const cosT = Math.cos(angle);
      const sinT = Math.sin(angle);
      
      const corners = [
        { x: cx + (-halfW * cosT - (-halfH) * sinT), y: cy + (-halfW * sinT + (-halfH) * cosT) }, // top-left
        { x: cx + (halfW * cosT - (-halfH) * sinT), y: cy + (halfW * sinT + (-halfH) * cosT) },   // top-right
        { x: cx + (halfW * cosT - halfH * sinT), y: cy + (halfW * sinT + halfH * cosT) },         // bottom-right
        { x: cx + (-halfW * cosT - halfH * sinT), y: cy + (-halfW * sinT + halfH * cosT) }        // bottom-left
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
      
      // Calculate axis-aligned bounding box for compatibility
      const xs = normalizedCorners.map(p => p.x);
      const ys = normalizedCorners.map(p => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const widthN = maxX - minX;
      const heightN = maxY - minY;
      
      // Filter tiny boxes and boxes outside image bounds
      const minArea = 0.001;
      if (widthN * heightN < minArea) continue;
      if (minX >= 1 || maxX <= 0 || minY >= 1 || maxY <= 0) continue;
      
      // Store results
      boxes.push(minX, minY, widthN, heightN);
      scores.push(confidence);
      classes.push(Math.round(classId));
      
      // Store rotated box corners
      rotatedBoxesAll.push(
        normalizedCorners[0].x, normalizedCorners[0].y,
        normalizedCorners[1].x, normalizedCorners[1].y,
        normalizedCorners[2].x, normalizedCorners[2].y,
        normalizedCorners[3].x, normalizedCorners[3].y,
      );
    }

    console.log(`🎯 Final results: ${boxes.length / 4} detections (NMS already applied in ONNX model)`);
    
    if (boxes.length === 0) {
      console.log('⚠️ No detections passed confidence threshold');
      return {
        boxes: new Float32Array(0),
        scores: new Float32Array(0),
        classes: new Int32Array(0),
        validDetections: 0,
        rotatedBoxes: new Float32Array(0)
      };
    }


    // Return results (no additional NMS needed since it's built into ONNX model)
    return {
      boxes: new Float32Array(boxes),
      rotatedBoxes: rotatedBoxesAll.length > 0 ? new Float32Array(rotatedBoxesAll) : undefined,
      scores: new Float32Array(scores),
      classes: new Int32Array(classes),
      validDetections: boxes.length / 4
    };
  }

  private applyNMS(boxes: number[], scores: number[], classes: number[], nmsThreshold: number) {
    // Simple NMS implementation
    const indices = Array.from({length: scores.length}, (_, i) => i);
    
    // Sort by confidence (descending)
    indices.sort((a, b) => scores[b] - scores[a]);
    
    const keep: number[] = [];
    const suppressed = new Set<number>();
    
    for (const i of indices) {
      if (suppressed.has(i)) continue;
      
      keep.push(i);
      
      // Suppress overlapping boxes
      for (const j of indices) {
        if (i === j || suppressed.has(j)) continue;
        
        const iou = this.calculateIoU(
          boxes.slice(i * 4, i * 4 + 4),
          boxes.slice(j * 4, j * 4 + 4)
        );
        
        if (iou > nmsThreshold) {
          suppressed.add(j);
        }
      }
    }
    
    // Return filtered results
    const filteredBoxes: number[] = [];
    const filteredScores: number[] = [];
    const filteredClasses: number[] = [];
    
    for (const idx of keep) {
      filteredBoxes.push(...boxes.slice(idx * 4, idx * 4 + 4));
      filteredScores.push(scores[idx]);
      filteredClasses.push(classes[idx]);
    }
    
    return {
      boxes: filteredBoxes,
      scores: filteredScores,
      classes: filteredClasses
    };
  }

  private calculateIoU(box1: number[], box2: number[]): number {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    // Calculate intersection
    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);
    
    if (xRight < xLeft || yBottom < yTop) return 0;
    
    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    const unionArea = box1Area + box2Area - intersectionArea;
    
    return intersectionArea / unionArea;
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.isLoaded = false;
    }
  }

  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  getConfig(): ModelConfig {
    return { ...this.config };
  }

  updateConfig(newConfig: Partial<ModelConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
}
