/**
 * Oriented Bounding Box (OBB) Non-Maximum Suppression Processor
 * 
 * Handles post-processing of YOLO OBB backbone model outputs:
 * Input: Raw model outputs (1, 6, 8400) - [cx, cy, w, h, angle, confidence]
 * Output: Filtered detections with NMS applied
 */

export interface OBBDetection {
  // Bounding box in normalized coordinates (0-1)
  boundingBox: {
    x: number;      // top-left x
    y: number;      // top-left y  
    width: number;  // width
    height: number; // height
  };
  // Oriented bounding box corners in normalized coordinates
  corners: {
    x1: number; y1: number; // top-left
    x2: number; y2: number; // top-right
    x3: number; y3: number; // bottom-right
    x4: number; y4: number; // bottom-left
  };
  confidence: number;
  classId: number;
  angle: number; // rotation angle in radians
}

export interface NMSConfig {
  confidenceThreshold: number;
  nmsThreshold: number;
  maxDetections: number;
  inputSize: number; // Model input size (e.g., 640)
}

export class OBBNMSProcessor {
  private config: NMSConfig;

  constructor(config: NMSConfig) {
    this.config = config;
  }

  /**
   * Process raw YOLO OBB model outputs
   * @param rawOutput Raw model output tensor data (1, 6, 8400)
   * @param imageWidth Original image width
   * @param imageHeight Original image height
   * @returns Filtered detections with NMS applied
   */
  processDetections(
    rawOutput: Float32Array,
    imageWidth: number,
    imageHeight: number
  ): OBBDetection[] {
    console.log('🔄 Processing OBB detections...');
    console.log(`📊 Raw output length: ${rawOutput.length}`);
    console.log(`📐 Image dimensions: ${imageWidth}x${imageHeight}`);
    console.log(`🔧 NMS Config:`, this.config);

    // Parse raw outputs - YOLO OBB format: [cx, cy, w, h, angle, confidence]
    const detections = this.parseRawDetections(rawOutput);
    console.log(`📊 Parsed detections: ${detections.length}`);

    // Filter by confidence threshold
    const confidenceFiltered = detections.filter(
      det => det.confidence >= this.config.confidenceThreshold
    );
    console.log(`📊 After confidence filtering (>${this.config.confidenceThreshold}): ${confidenceFiltered.length}`);
    
    // Show confidence distribution
    const confidences = detections.map(d => d.confidence).sort((a, b) => b - a);
    console.log(`📊 Top 10 confidence scores: [${confidences.slice(0, 10).map(c => c.toFixed(3)).join(', ')}]`);
    console.log(`📊 Confidence range: ${confidences[0]?.toFixed(3)} - ${confidences[confidences.length-1]?.toFixed(3)}`);

    if (confidenceFiltered.length === 0) {
      return [];
    }

    // Convert to image coordinates and create oriented boxes
    const imageDetections = confidenceFiltered.map(det => 
      this.convertToImageCoordinates(det, imageWidth, imageHeight)
    );

    // Apply Non-Maximum Suppression
    const nmsFiltered = this.applyNMS(imageDetections);
    console.log(`📊 After NMS (threshold: ${this.config.nmsThreshold}): ${nmsFiltered.length}`);

    // Limit to max detections
    const finalDetections = nmsFiltered
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, this.config.maxDetections);

    console.log(`✅ Final detections: ${finalDetections.length}`);
    return finalDetections;
  }

  /**
   * Parse raw model outputs - coordinates are ALREADY DECODED, only confidence needs sigmoid
   */
  private parseRawDetections(rawOutput: Float32Array): Array<{
    cx: number; cy: number; w: number; h: number; angle: number; confidence: number;
  }> {
    const detections: Array<{
      cx: number; cy: number; w: number; h: number; angle: number; confidence: number;
    }> = [];

    // YOLO OBB output format: (1, 6, N) -> 6 channels × N anchors
    // Calculate actual number of anchors from data length
    const numChannels = 6;
    const numAnchors = rawOutput.length / numChannels;

    console.log(`🔍 Parsing backbone model outputs: ${rawOutput.length} values, expected: ${numChannels * numAnchors}`);

    // Extract channel data (channel-first format)
    const chStride = numAnchors;
    const cxRaw = rawOutput.subarray(0 * chStride, 1 * chStride);
    const cyRaw = rawOutput.subarray(1 * chStride, 2 * chStride);
    const wRaw = rawOutput.subarray(2 * chStride, 3 * chStride);
    const hRaw = rawOutput.subarray(3 * chStride, 4 * chStride);
    const thRaw = rawOutput.subarray(4 * chStride, 5 * chStride);
    const scRaw = rawOutput.subarray(5 * chStride, 6 * chStride);

    console.log(`📊 Coordinate ranges: cx[${cxRaw[0]?.toFixed(1)}-${Math.max.apply(null, Array.from(cxRaw)).toFixed(1)}], cy[${cyRaw[0]?.toFixed(1)}-${Math.max.apply(null, Array.from(cyRaw)).toFixed(1)}]`);
    console.log(`📊 Confidence range: [${Math.min.apply(null, Array.from(scRaw)).toFixed(3)}-${Math.max.apply(null, Array.from(scRaw)).toFixed(3)}]`);

    // Process all anchors - coordinates are already decoded to pixel space
    for (let i = 0; i < numAnchors; i++) {
      // Coordinates are already in model input pixel space - NO DECODING NEEDED
      const cx = cxRaw[i];
      const cy = cyRaw[i];
      const w = wRaw[i];
      const h = hRaw[i];
      // Angle processing - the values seem to be in [0,1] range, need to convert to radians
      // Assuming they represent normalized angles that need to be scaled to [0, 2π]
      const angleRaw = thRaw[i];
      const angle = angleRaw * 2 * Math.PI; // Convert from [0,1] to [0,2π] radians
      const confidence = this.sigmoid(scRaw[i]); // ONLY confidence needs sigmoid

      detections.push({ cx, cy, w, h, angle, confidence });
    }

    console.log(`📊 Parsed ${detections.length} detections (coordinates already decoded)`);
    return detections;
  }

  /**
   * Convert model coordinates to image coordinates with letterbox correction
   */
  private convertToImageCoordinates(
    detection: { cx: number; cy: number; w: number; h: number; angle: number; confidence: number },
    imageWidth: number,
    imageHeight: number
  ): OBBDetection {
    const { cx, cy, w, h, angle, confidence } = detection;

    // Calculate letterbox parameters EXACTLY matching frontend preprocessing
    const scale = Math.min(
      this.config.inputSize / imageWidth,
      this.config.inputSize / imageHeight
    );
    const newWidth = Math.round(imageWidth * scale);
    const newHeight = Math.round(imageHeight * scale);
    const padX = Math.floor((this.config.inputSize - newWidth) / 2);
    const padY = Math.floor((this.config.inputSize - newHeight) / 2);

    // Convert from model coordinates to original image coordinates
    const imgCx = (cx - padX) / scale;
    const imgCy = (cy - padY) / scale;
    const imgW = w / scale;
    const imgH = h / scale;

    // Debug coordinate conversion
    if (Math.random() < 0.01) { // Log 1% of detections for debugging
      console.log('🔍 Coordinate conversion debug:', {
        model: { cx, cy, w, h, angle, confidence },
        letterbox: { scale, newWidth, newHeight, padX, padY },
        image: { imgCx, imgCy, imgW, imgH },
        imageSize: { imageWidth, imageHeight },
        inputSize: this.config.inputSize
      });
    }

    // Calculate oriented bounding box corners
    const corners = this.calculateOBBCorners(imgCx, imgCy, imgW, imgH, angle);

    // Calculate axis-aligned bounding box for NMS
    const xs = [corners.x1, corners.x2, corners.x3, corners.x4];
    const ys = [corners.y1, corners.y2, corners.y3, corners.y4];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    // Normalize coordinates to [0, 1]
    const boundingBox = {
      x: Math.max(0, Math.min(1, minX / imageWidth)),
      y: Math.max(0, Math.min(1, minY / imageHeight)),
      width: Math.max(0, Math.min(1, (maxX - minX) / imageWidth)),
      height: Math.max(0, Math.min(1, (maxY - minY) / imageHeight))
    };

    // Normalize corner coordinates
    const normalizedCorners = {
      x1: Math.max(0, Math.min(1, corners.x1 / imageWidth)),
      y1: Math.max(0, Math.min(1, corners.y1 / imageHeight)),
      x2: Math.max(0, Math.min(1, corners.x2 / imageWidth)),
      y2: Math.max(0, Math.min(1, corners.y2 / imageHeight)),
      x3: Math.max(0, Math.min(1, corners.x3 / imageWidth)),
      y3: Math.max(0, Math.min(1, corners.y3 / imageHeight)),
      x4: Math.max(0, Math.min(1, corners.x4 / imageWidth)),
      y4: Math.max(0, Math.min(1, corners.y4 / imageHeight))
    };

    return {
      boundingBox,
      corners: normalizedCorners,
      confidence,
      classId: 0, // Single class (trading card)
      angle
    };
  }

  /**
   * Calculate oriented bounding box corners from center, size, and angle
   */
  private calculateOBBCorners(
    cx: number, cy: number, w: number, h: number, angle: number
  ): { x1: number; y1: number; x2: number; y2: number; x3: number; y3: number; x4: number; y4: number } {
    const halfW = w / 2;
    const halfH = h / 2;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);

    // Corner offsets relative to center (before rotation)
    const corners = [
      [-halfW, -halfH], // top-left
      [halfW, -halfH],  // top-right
      [halfW, halfH],   // bottom-right
      [-halfW, halfH]   // bottom-left
    ];

    // Apply rotation and translation
    const rotatedCorners = corners.map(([x, y]) => ({
      x: cx + x * cosA - y * sinA,
      y: cy + x * sinA + y * cosA
    }));

    return {
      x1: rotatedCorners[0].x, y1: rotatedCorners[0].y,
      x2: rotatedCorners[1].x, y2: rotatedCorners[1].y,
      x3: rotatedCorners[2].x, y3: rotatedCorners[2].y,
      x4: rotatedCorners[3].x, y4: rotatedCorners[3].y
    };
  }

  /**
   * Apply Non-Maximum Suppression to filter overlapping detections
   */
  private applyNMS(detections: OBBDetection[]): OBBDetection[] {
    if (detections.length === 0) return [];

    // Sort by confidence (descending)
    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    
    const keep: OBBDetection[] = [];
    const suppressed = new Set<number>();

    for (let i = 0; i < sorted.length; i++) {
      if (suppressed.has(i)) continue;

      const currentDetection = sorted[i];
      keep.push(currentDetection);

      // Suppress overlapping detections
      for (let j = i + 1; j < sorted.length; j++) {
        if (suppressed.has(j)) continue;

        const otherDetection = sorted[j];
        const iou = this.calculateIoU(currentDetection.boundingBox, otherDetection.boundingBox);

        if (iou > this.config.nmsThreshold) {
          suppressed.add(j);
        }
      }
    }

    return keep;
  }

  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   */
  private calculateIoU(
    box1: { x: number; y: number; width: number; height: number },
    box2: { x: number; y: number; width: number; height: number }
  ): number {
    // Calculate intersection
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    if (x2 < x1 || y2 < y1) return 0;

    const intersectionArea = (x2 - x1) * (y2 - y1);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const unionArea = box1Area + box2Area - intersectionArea;

    return unionArea > 0 ? intersectionArea / unionArea : 0;
  }

  /**
   * Sigmoid activation function
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Update NMS configuration
   */
  updateConfig(newConfig: Partial<NMSConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): NMSConfig {
    return { ...this.config };
  }
}
