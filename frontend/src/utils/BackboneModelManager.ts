import * as ort from 'onnxruntime-web';
import { OBBNMSProcessor, OBBDetection, NMSConfig } from './OBBNMSProcessor';

export interface BackboneModelConfig {
  modelPath: string;
  inputSize: number; // Model input size (e.g., 640)
  executionProviders: string[];
  nmsConfig: NMSConfig;
}

export interface BackboneModelPrediction {
  detections: OBBDetection[];
  inferenceTime: number;
  processingTime: number;
}

/**
 * Model manager for YOLO OBB backbone model with JavaScript NMS post-processing
 * Handles the complete pipeline: backbone model → NMS → final detections
 */
export class BackboneModelManager {
  private session: ort.InferenceSession | null = null;
  private config: BackboneModelConfig;
  private nmsProcessor: OBBNMSProcessor;
  private isLoaded: boolean = false;

  constructor(config: BackboneModelConfig) {
    this.config = config;
    this.nmsProcessor = new OBBNMSProcessor(config.nmsConfig);
    this.setupONNXRuntime();
  }

  private setupONNXRuntime(): void {
    // Point to local ONNX Runtime assets
    const base = (process.env.PUBLIC_URL || '') + '/onnx/';
    ort.env.wasm.wasmPaths = base;
    
    // Optimize WASM configuration for performance
    const maxThreads = Math.min(navigator.hardwareConcurrency || 4, 8);
    ort.env.wasm.numThreads = maxThreads;
    ort.env.wasm.simd = true; // Enable SIMD for significant performance boost
    ort.env.wasm.proxy = false; // Avoid proxy loader issues
    
    // Try to set more permissive settings for opset compatibility
    try {
      // Set environment variables for better compatibility
      (ort.env as any).debug = false;
      (ort.env as any).logLevel = 'warning';
    } catch (error) {
      console.log('⚠️ Could not set additional env settings:', error);
    }
    
    console.log('🔧 Backbone Model Manager - ONNX Runtime configured for maximum compatibility');
    console.log('🔧 ONNX Runtime optimized configuration:');
    console.log(`   - WASM path: ${ort.env.wasm.wasmPaths}`);
    console.log(`   - Threads: ${ort.env.wasm.numThreads} (max available: ${navigator.hardwareConcurrency})`);
    console.log(`   - SIMD enabled: ${ort.env.wasm.simd}`);
    console.log(`   - WebGPU available: ${this.isWebGPUSupported()}`);
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

    try {
      console.log('🚀 Loading backbone model from:', this.config.modelPath);
      console.log('📊 Execution providers:', this.config.executionProviders);
      
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: this.config.executionProviders,
        graphOptimizationLevel: 'all', // Enable all optimizations for better performance
        enableMemPattern: true, // Enable memory pattern optimization
        enableCpuMemArena: true, // Enable CPU memory arena for better memory management
        executionMode: 'parallel', // Enable parallel execution
        interOpNumThreads: Math.min(navigator.hardwareConcurrency || 4, 8),
        intraOpNumThreads: Math.min(navigator.hardwareConcurrency || 4, 8),
        logSeverityLevel: 2, // Reduce logging for better performance (2 = warning level)
        logVerbosityLevel: 0, // Minimal verbosity for performance
      };

      const loadStartTime = performance.now();
      this.session = await ort.InferenceSession.create(
        this.config.modelPath,
        sessionOptions
      );
      const loadTime = performance.now() - loadStartTime;

      this.isLoaded = true;
      console.log(`✅ Backbone model loaded successfully in ${loadTime.toFixed(2)}ms!`);
      console.log('📥 Input names:', this.session.inputNames);
      console.log('📤 Output names:', this.session.outputNames);
      
      // Warm up the model with a dummy inference
      await this.warmUpModel();
      
      console.log('🎯 Backbone model ready for high-performance inference');
      console.log('🎯 Expected output format: (1, 6, 8400) - [cx, cy, w, h, angle, confidence]');

    } catch (error) {
      console.error('❌ Backbone model loading failed:', error);
      throw new Error(`Backbone model loading failed: ${error}`);
    }
  }

  private async warmUpModel(): Promise<void> {
    if (!this.session) return;
    
    try {
      console.log('🔥 Warming up backbone model for optimal performance...');
      const warmupStartTime = performance.now();
      
      // Create a dummy input tensor with the correct shape
      const inputSize = this.config.inputSize;
      const dummyData = new Float32Array(1 * 3 * inputSize * inputSize);
      // Fill with random-ish data to simulate real input
      for (let i = 0; i < dummyData.length; i++) {
        dummyData[i] = Math.random() * 0.5 + 0.25; // Values between 0.25 and 0.75
      }
      
      const dummyTensor = new ort.Tensor('float32', dummyData, [1, 3, inputSize, inputSize]);
      
      // Run a few warmup inferences
      for (let i = 0; i < 3; i++) {
        await this.session.run({
          [this.session.inputNames[0]]: dummyTensor
        });
      }
      
      const warmupTime = performance.now() - warmupStartTime;
      console.log(`🔥 Backbone model warmup completed in ${warmupTime.toFixed(2)}ms`);
      console.log('⚡ Backbone model is now optimized for fast inference');
      
    } catch (error) {
      console.warn('⚠️ Backbone model warmup failed (non-critical):', error);
    }
  }

  async predict(imageData: ImageData): Promise<BackboneModelPrediction> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    try {
      const startTime = performance.now();

      // Preprocess image
      const inputTensor = this.preprocessImage(imageData);
      
      // Run backbone model inference
      const inferenceStartTime = performance.now();
      const results = await this.session.run({
        [this.session.inputNames[0]]: inputTensor
      });
      const inferenceTime = performance.now() - inferenceStartTime;
      
      console.log(`🔥 Backbone inference completed in ${inferenceTime.toFixed(2)}ms`);

      // Post-process with JavaScript NMS
      const processingStartTime = performance.now();
      const detections = await this.postprocessResults(results, imageData.width, imageData.height);
      const processingTime = performance.now() - processingStartTime;

      const totalTime = performance.now() - startTime;
      console.log(`⚡ Total pipeline time: ${totalTime.toFixed(2)}ms (inference: ${inferenceTime.toFixed(2)}ms, processing: ${processingTime.toFixed(2)}ms)`);

      return {
        detections,
        inferenceTime,
        processingTime
      };

    } catch (error) {
      console.error('❌ Prediction failed:', error);
      throw new Error(`Prediction failed: ${error}`);
    }
  }

  private preprocessImage(imageData: ImageData): ort.Tensor {
    const { width, height, data } = imageData;
    const inputSize = this.config.inputSize;

    console.log(`🔄 Preprocessing image: ${width}x${height} → ${inputSize}x${inputSize}`);

    // Letterbox resize to maintain aspect ratio
    const scale = Math.min(inputSize / width, inputSize / height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    const padX = Math.floor((inputSize - newWidth) / 2);
    const padY = Math.floor((inputSize - newHeight) / 2);

    console.log(`📐 Letterbox params: scale=${scale.toFixed(3)}, newSize=${newWidth}x${newHeight}, pad=${padX},${padY}`);

    // Create source canvas
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCtx.putImageData(new ImageData(data, width, height), 0, 0);

    // Create destination canvas with letterbox
    const dstCanvas = document.createElement('canvas');
    const dstCtx = dstCanvas.getContext('2d')!;
    dstCanvas.width = inputSize;
    dstCanvas.height = inputSize;
    
    // Fill with gray background (matching YOLO preprocessing)
    dstCtx.fillStyle = '#808080';
    dstCtx.fillRect(0, 0, inputSize, inputSize);
    
    // Draw resized image
    dstCtx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newWidth, newHeight);

    // Convert to tensor format: [1, 3, H, W] with values normalized to [0, 1]
    const resizedImageData = dstCtx.getImageData(0, 0, inputSize, inputSize);
    const tensorData = new Float32Array(1 * 3 * inputSize * inputSize);
    const pixels = resizedImageData.data;

    for (let i = 0; i < inputSize * inputSize; i++) {
      const pixelIndex = i * 4;
      const tensorIndex = i;
      
      // RGB channels: [R, G, B] normalized to [0, 1] (standard format for YOLO11n)
      tensorData[tensorIndex] = pixels[pixelIndex] / 255.0;     // R
      tensorData[tensorIndex + inputSize * inputSize] = pixels[pixelIndex + 1] / 255.0; // G
      tensorData[tensorIndex + 2 * inputSize * inputSize] = pixels[pixelIndex + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]);
  }

  private async postprocessResults(
    results: ort.InferenceSession.ReturnType, 
    imageWidth: number, 
    imageHeight: number
  ): Promise<OBBDetection[]> {
    // Extract model output
    const outputNames = Object.keys(results);
    const output = results[outputNames[0]];
    
    if (!output) {
      throw new Error('Invalid model output');
    }

    console.log('📊 Raw model output shape:', (output as any).dims);

    // Download GPU data to CPU if needed
    let outputData: Float32Array;
    if ((output as any).location === 'gpu-buffer') {
      console.log('📥 Downloading GPU data to CPU...');
      outputData = await (output as any).getData() as Float32Array;
    } else {
      outputData = (output as any).data as Float32Array;
    }

    if (!outputData) {
      throw new Error('Failed to get output data from model');
    }

    console.log(`📊 Output data length: ${outputData.length}`);
    console.log('📊 Expected format: (1, 6, 8400) = 50,400 values');
    console.log('📊 First 10 values:', Array.from(outputData.slice(0, 10)));
    console.log('📊 Sample values from each channel:');
    console.log('  - cx (0-10):', Array.from(outputData.slice(0, 10)));
    console.log('  - cy (8400-8410):', Array.from(outputData.slice(8400, 8410)));
    console.log('  - w (16800-16810):', Array.from(outputData.slice(16800, 16810)));
    console.log('  - h (25200-25210):', Array.from(outputData.slice(25200, 25210)));
    console.log('  - angle (33600-33610):', Array.from(outputData.slice(33600, 33610)));
    console.log('  - conf (42000-42010):', Array.from(outputData.slice(42000, 42010)));

    // Validate output format
    const expectedLength = 6 * 8400; // 6 channels × 8400 anchors
    if (outputData.length !== expectedLength) {
      console.warn(`⚠️ Unexpected output length: ${outputData.length}, expected: ${expectedLength}`);
    }

    // Process detections using JavaScript NMS
    const detections = this.nmsProcessor.processDetections(
      outputData,
      imageWidth,
      imageHeight
    );

    return detections;
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.isLoaded = false;
      console.log('🗑️ Backbone model disposed');
    }
  }

  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  getConfig(): BackboneModelConfig {
    return { ...this.config };
  }

  updateConfig(newConfig: Partial<BackboneModelConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update NMS processor if NMS config changed
    if (newConfig.nmsConfig) {
      this.nmsProcessor.updateConfig(newConfig.nmsConfig);
    }
  }

  updateNMSConfig(nmsConfig: Partial<NMSConfig>): void {
    this.nmsProcessor.updateConfig(nmsConfig);
    this.config.nmsConfig = { ...this.config.nmsConfig, ...nmsConfig };
  }

  getNMSConfig(): NMSConfig {
    return this.nmsProcessor.getConfig();
  }
}
