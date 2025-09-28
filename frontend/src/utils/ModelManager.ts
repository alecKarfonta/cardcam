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
    const features = dims && dims.length >= 3 ? dims[1] : 6;
    const numAnchors = dims && dims.length >= 3 ? dims[2] : Math.floor(outputData.length / 6);
    const expectedLength = numAnchors * features;

    console.log('📊 Output data length:', outputData.length);
    console.log(`📊 Expected length for ${numAnchors} anchors × ${features} features:`, expectedLength);
    console.log('📊 Output shape:', dims);
    console.log('📊 First 20 values:', Array.from(outputData.slice(0, 20)));

    const [, , targetHeight, targetWidth] = this.config.inputShape; // [1,3,640,640]

    const boxes: number[] = [];
    const scores: number[] = [];
    const classes: number[] = [];
    const rotatedBoxesAll: number[] = [];

    // helper: sigmoid
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

      if (features === 6) {
        // OBB model outputs: [cx, cy, w, h, angle, conf] - coordinates already in pixel space
        const chStride = numAnchors; // per-channel span
        const cxRaw = outputData.subarray(0 * chStride, 1 * chStride);
        const cyRaw = outputData.subarray(1 * chStride, 2 * chStride);
        const wRaw  = outputData.subarray(2 * chStride, 3 * chStride);
        const hRaw  = outputData.subarray(3 * chStride, 4 * chStride);
        const thRaw = outputData.subarray(4 * chStride, 5 * chStride);
        const scRaw = outputData.subarray(5 * chStride, 6 * chStride);

        console.log(`🔧 OBB decode for ${targetWidth}x${targetWidth}, ${numAnchors} anchors`);
        console.log(`📊 Raw ranges: cx[${cxRaw[0]?.toFixed(2)}-${Array.from(cxRaw).reduce((a,b) => Math.max(a,b), -Infinity).toFixed(2)}], cy[${cyRaw[0]?.toFixed(2)}-${Array.from(cyRaw).reduce((a,b) => Math.max(a,b), -Infinity).toFixed(2)}], conf[${Array.from(scRaw).reduce((a,b) => Math.min(a,b), Infinity).toFixed(6)}-${Array.from(scRaw).reduce((a,b) => Math.max(a,b), -Infinity).toFixed(6)}]`);
        
        for (let i = 0; i < numAnchors; i++) {
          // Coordinates are in pixel space relative to model input (1088x1088)
          const cx = cxRaw[i];
          const cy = cyRaw[i];
          const ww = wRaw[i];
          const hh = hRaw[i];
          const th = thRaw[i]; // Angle in radians
          const score = sigmoid(scRaw[i]); // Apply sigmoid to confidence
          
          if (score <= this.config.confidenceThreshold) continue;

          // Create rotated corners in letterbox space
          const halfW = ww / 2;
          const halfH = hh / 2;
          const cosT = Math.cos(th);
          const sinT = Math.sin(th);
          const rel = [
            [-halfW, -halfH],
            [ halfW, -halfH],
            [ halfW,  halfH],
            [-halfW,  halfH],
          ];
          const corners = rel.map(([rx, ry]) => ({
            x: cx + rx * cosT - ry * sinT,
            y: cy + rx * sinT + ry * cosT,
          }));
          
          // Map back from letterbox to original image then normalize
          const unpad = corners.map(p => ({
            x: (p.x - this.lastPadX) / this.lastScale,
            y: (p.y - this.lastPadY) / this.lastScale,
          }));
          const norm = unpad.map(p => ({
            x: Math.max(0, Math.min(1, p.x / this.lastOrigW)),
            y: Math.max(0, Math.min(1, p.y / this.lastOrigH)),
          }));
          
          // Calculate AABB for NMS
          const xs = norm.map(p => p.x);
          const ys = norm.map(p => p.y);
          const minX = Math.max(0, Math.min(...xs));
          const maxX = Math.min(1, Math.max(...xs));
          const minY = Math.max(0, Math.min(...ys));
          const maxY = Math.min(1, Math.max(...ys));
          const widthN  = Math.max(0, maxX - minX);
          const heightN = Math.max(0, maxY - minY);
          
          // Filter tiny boxes and boxes outside image bounds
          const minArea = 0.001;
          if (widthN * heightN < minArea) continue;
          if (minX >= 1 || maxX <= 0 || minY >= 1 || maxY <= 0) continue;

          boxes.push(minX, minY, widthN, heightN);
          scores.push(score);
          classes.push(0);
          rotatedBoxesAll.push(
            norm[0].x, norm[0].y,
            norm[1].x, norm[1].y,
            norm[2].x, norm[2].y,
            norm[3].x, norm[3].y,
          );
        }
    } else {
      // Fallback: treat as axis-aligned [cx,cy,w,h,obj,cls]
      const idxCx = 0 * numAnchors;
      const idxCy = 1 * numAnchors;
      const idxW  = 2 * numAnchors;
      const idxH  = 3 * numAnchors;
      const idxObj= 4 * numAnchors;
      const idxCls= 5 * numAnchors;

      for (let i = 0; i < numAnchors; i++) {
        const cx = outputData[idxCx + i];
        const cy = outputData[idxCy + i];
        const w  = outputData[idxW  + i];
        const h  = outputData[idxH  + i];
        const obj= sigmoid(outputData[idxObj+ i]);
        const cls= sigmoid(outputData[idxCls+ i]);
        const score = (obj || 0) * (cls || 0);
        if (score <= this.config.confidenceThreshold) continue;

          const xPix = (cx - w / 2 - this.lastPadX) / this.lastScale;
          const yPix = (cy - h / 2 - this.lastPadY) / this.lastScale;
          const wPix = w / this.lastScale;
          const hPix = h / this.lastScale;
          const x = Math.max(0, xPix / this.lastOrigW);
          const y = Math.max(0, yPix / this.lastOrigH);
          const widthN  = Math.min(1, Math.max(0, wPix / this.lastOrigW));
          const heightN = Math.min(1, Math.max(0, hPix / this.lastOrigH));
        const minArea = 0.001;
        if (widthN * heightN < minArea) continue;

        boxes.push(x, y, widthN, heightN);
        scores.push(score);
        classes.push(0);
      }
    }

    console.log(`🎯 Post-processing: ${boxes.length / 4} boxes before NMS (conf > ${this.config.confidenceThreshold})`);
    if (boxes.length === 0) {
      console.log('⚠️ No boxes passed confidence threshold');
      return {
        boxes: new Float32Array(0),
        scores: new Float32Array(0),
        classes: new Int32Array(0),
        validDetections: 0,
        rotatedBoxes: new Float32Array(0)
      };
    }

    // Show top detections for debugging
    const topDetections = scores.map((score, i) => ({ score, i }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    console.log('🔝 Top 5 detections:', topDetections.map(d => 
      `score=${d.score.toFixed(3)}, box=[${boxes[d.i*4].toFixed(3)}, ${boxes[d.i*4+1].toFixed(3)}, ${boxes[d.i*4+2].toFixed(3)}, ${boxes[d.i*4+3].toFixed(3)}]`
    ));

    // Keep top-K before NMS to reduce clutter
    const topK = 50;
    if (scores.length > topK) {
      const idx = scores.map((s, i) => [s, i] as [number, number]).sort((a,b) => b[0]-a[0]).slice(0, topK).map(x => x[1]);
      const newBoxes: number[] = [];
      const newScores: number[] = [];
      const newClasses: number[] = [];
      for (const i of idx) {
        newBoxes.push(...boxes.slice(i*4, i*4+4));
        newScores.push(scores[i]);
        newClasses.push(classes[i]);
      }
      boxes.length = 0; boxes.push(...newBoxes);
      scores.length = 0; scores.push(...newScores);
      classes.length = 0; classes.push(...newClasses);
    }

    console.log(`📊 Detections above confidence threshold (${this.config.confidenceThreshold}): ${scores.length}`);
    console.log('📊 Confidence scores:', scores.slice(0, 10));

    // NMS on normalized axis-aligned boxes (approx for rotated)
    const nmsResults = this.applyNMS(boxes, scores, classes, this.config.nmsThreshold);
    console.log(`📊 Detections after NMS: ${nmsResults.scores.length}`);

    // If we have rotated boxes, filter them by NMS keep set
    let rotatedOut: number[] | undefined = undefined;
    if (rotatedBoxesAll.length > 0) {
      const kept: number[] = [];
      // Build mapping by matching scores and AABBs in order; since we kept top-K and NMS sorted, use a tolerant match
      const keepSet: number[] = [];
      // Recompute original arrays for matching
      const origBoxes = boxes;
      const origScores = scores;
      for (let i = 0; i < nmsResults.boxes.length / 4; i++) {
        const kb = nmsResults.boxes.slice(i*4, i*4+4);
        const ks = nmsResults.scores[i];
        for (let j = 0; j < origScores.length; j++) {
          if (keepSet.includes(j)) continue;
          const ob = origBoxes.slice(j*4, j*4+4);
          const os = origScores[j];
          if (Math.abs(os - ks) < 1e-6 && Math.abs(ob[0]-kb[0])<1e-6 && Math.abs(ob[1]-kb[1])<1e-6) {
            kept.push(j);
            keepSet.push(j);
            break;
          }
        }
      }
      rotatedOut = [];
      for (const j of kept) {
        rotatedOut.push(...rotatedBoxesAll.slice(j*8, j*8+8));
      }
    }

    return {
      boxes: new Float32Array(nmsResults.boxes),
      rotatedBoxes: rotatedOut ? new Float32Array(rotatedOut) : undefined,
      scores: new Float32Array(nmsResults.scores),
      classes: new Int32Array(nmsResults.classes),
      validDetections: nmsResults.boxes.length / 4
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
