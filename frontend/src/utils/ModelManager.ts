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
    // Point to local ONNX Runtime assets that match the installed package version
    const base = (process.env.PUBLIC_URL || '') + '/onnx/';
    ort.env.wasm.wasmPaths = base;
    ort.env.wasm.numThreads = 1; // Single threaded to avoid issues
    ort.env.wasm.simd = false; // Avoid SIMD/threaded variants to reduce dynamic imports
    ort.env.wasm.proxy = false; // Avoid proxy loader that uses .mjs dynamic imports
    
    console.log('🔧 ONNX Runtime configured with local WASM path:', ort.env.wasm.wasmPaths);
    console.log('🔧 Using single-threaded WASM for compatibility');
    console.log('🔧 SIMD disabled to avoid MIME type issues');
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
      console.log('🚀 Starting model load from:', this.config.modelPath);
      console.log('📊 Execution providers:', this.config.executionProviders);
      console.log('🔧 ONNX Runtime environment:', {
        wasmPaths: ort.env.wasm.wasmPaths,
        numThreads: ort.env.wasm.numThreads,
        simd: ort.env.wasm.simd,
        proxy: ort.env.wasm.proxy
      });
      
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: this.config.executionProviders,
        graphOptimizationLevel: 'basic', // Use basic optimization to avoid issues
        enableMemPattern: false, // Disable to avoid potential issues
        enableCpuMemArena: false, // Disable to avoid potential issues
      };

      console.log('📋 Session options:', sessionOptions);

      this.session = await ort.InferenceSession.create(
        this.config.modelPath,
        sessionOptions
      );

      this.isLoaded = true;
      console.log('✅ Model loaded successfully!');
      console.log('📥 Input names:', this.session.inputNames);
      console.log('📤 Output names:', this.session.outputNames);
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

    try {
      // Preprocess image
      const inputTensor = this.preprocessImage(imageData);
      
      // Run inference
      const startTime = performance.now();
      const results = await this.session.run({
        [this.session.inputNames[0]]: inputTensor
      });
      const inferenceTime = performance.now() - startTime;
      
      console.log(`Inference completed in ${inferenceTime.toFixed(2)}ms`);

      // Post-process results
      return await this.postprocessResults(results);

    } catch (error) {
      console.error('Prediction failed:', error);
      throw new Error(`Prediction failed: ${error}`);
    }
  }

  private preprocessImage(imageData: ImageData): ort.Tensor {
    const { width, height, data } = imageData;
    const [, , targetH, targetW] = this.config.inputShape; // now 1088x1088

    // Letterbox to keep aspect ratio like Ultralytics
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

    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = width;
    srcCanvas.height = height;
    srcCtx.putImageData(new ImageData(data, width, height), 0, 0);

    const dstCanvas = document.createElement('canvas');
    const dstCtx = dstCanvas.getContext('2d')!;
    dstCanvas.width = targetW;
    dstCanvas.height = targetH;
    // fill pad with gray (matching Ultralytics default)
    dstCtx.fillStyle = '#808080';
    dstCtx.fillRect(0, 0, targetW, targetH);
    dstCtx.drawImage(srcCanvas, 0, 0, width, height, padX, padY, newW, newH);

    const resized = dstCtx.getImageData(0, 0, targetW, targetH);
    const tensorData = new Float32Array(1 * 3 * targetH * targetW);
    const pixels = resized.data;
    for (let i = 0; i < targetH * targetW; i++) {
      const p = i * 4;
      tensorData[i] = pixels[p] / 255.0;
      tensorData[i + targetH * targetW] = pixels[p + 1] / 255.0;
      tensorData[i + 2 * targetH * targetW] = pixels[p + 2] / 255.0;
    }
    return new ort.Tensor('float32', tensorData, this.config.inputShape);
  }

  private async postprocessResults(results: ort.InferenceSession.ReturnType): Promise<ModelPrediction> {
    // Extract outputs
    const outputNames = Object.keys(results);
    const output = results[outputNames[0]];
    if (!output) {
      throw new Error('Invalid model output');
    }

    // Download GPU data to CPU if needed
    let outputData: Float32Array;
    if ((output as any).location === 'gpu-buffer') {
      console.log('📥 Downloading GPU data to CPU...');
      outputData = await (output as any).getData() as Float32Array;
    } else {
      outputData = (output as any).data as Float32Array;
    }

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

    // Show detections for debugging
    console.log('🔝 Detections:', scores.map((score, i) => 
      `score=${score.toFixed(3)}, box=[${boxes[i*4].toFixed(3)}, ${boxes[i*4+1].toFixed(3)}, ${boxes[i*4+2].toFixed(3)}, ${boxes[i*4+3].toFixed(3)}]`
    ));

    console.log(`📊 Confidence scores:`, scores);

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
