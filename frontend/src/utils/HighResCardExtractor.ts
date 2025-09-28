import { CardDetection } from '../store/slices/inferenceSlice';

export interface HighResExtractionOptions {
  modelInputSize: { width: number; height: number };
  paddingRatio: number;
  minCardSize: number;
  maxCardSize: number;
  enablePerspectiveCorrection: boolean;
}

export interface HighResExtractionResult {
  imageData: ImageData;
  extractedWidth: number;
  extractedHeight: number;
  metadata: {
    cardId: string;
    extractionMethod: 'bbox' | 'obb' | 'perspective';
    originalSize: { width: number; height: number };
    modelInputSize: { width: number; height: number };
    scalingFactors: { x: number; y: number };
    confidence: number;
    rotationAngle?: number;
    corners?: Array<{ x: number; y: number }>;
    paddingApplied: { x: number; y: number };
    isHighResolution: boolean;
  };
}

/**
 * High-Resolution Card Extractor
 * 
 * Extracts individual trading cards from original full-resolution images using detection
 * coordinates from models that were run on downsampled inputs. Handles coordinate
 * transformation, oriented bounding box extraction, and perspective correction.
 */
export class HighResCardExtractor {
  private options: HighResExtractionOptions;

  constructor(options: Partial<HighResExtractionOptions> = {}) {
    this.options = {
      modelInputSize: { width: 1088, height: 1088 }, // Actual model input size (square)
      paddingRatio: 0.05, // 5% padding around detected cards
      minCardSize: 200,
      maxCardSize: 2000,
      enablePerspectiveCorrection: true,
      ...options
    };
  }

  /**
   * Extract cards from original full-resolution image using detection results
   */
  async extractCards(
    originalImage: HTMLImageElement | HTMLCanvasElement | ImageData,
    detections: CardDetection[],
    originalImageSize?: { width: number; height: number }
  ): Promise<HighResExtractionResult[]> {
    if (detections.length === 0) {
      console.log('🚫 No detections provided for extraction');
      return [];
    }

    // Get original image data
    const originalImageData = await this.getImageData(originalImage, originalImageSize);
    if (!originalImageData) {
      throw new Error('Failed to get image data from original image');
    }

    console.log('🎯 HIGH-RESOLUTION EXTRACTION STARTING');
    console.log(`📊 Extracting ${detections.length} cards from ${originalImageData.width}x${originalImageData.height} image`);
    console.log(`🤖 Model input size: ${this.options.modelInputSize.width}x${this.options.modelInputSize.height}`);
    console.log(`📏 Padding ratio: ${this.options.paddingRatio} (${(this.options.paddingRatio * 100).toFixed(1)}%)`);

    // Calculate scaling factors from model input to original image
    const scalingFactors = {
      x: originalImageData.width / this.options.modelInputSize.width,
      y: originalImageData.height / this.options.modelInputSize.height
    };

    console.log(`📐 Scaling factors: x=${scalingFactors.x.toFixed(3)}, y=${scalingFactors.y.toFixed(3)}`);
    console.log(`🔍 Expected resolution improvement: ${(scalingFactors.x * scalingFactors.y).toFixed(1)}x more pixels`);

    const results: HighResExtractionResult[] = [];

    for (let i = 0; i < detections.length; i++) {
      const detection = detections[i];
      console.log(`\n🎴 Processing card ${i + 1}/${detections.length} (ID: ${detection.id})`);
      console.log(`   Confidence: ${(detection.confidence * 100).toFixed(1)}%`);
      console.log(`   Rotated: ${detection.isRotated ? 'Yes' : 'No'}`);
      console.log(`   Corners: ${detection.corners ? detection.corners.length : 'None'}`);
      console.log(`   BBox: ${JSON.stringify(detection.boundingBox)}`);
      
      try {
        const result = await this.extractSingleCard(
          originalImageData,
          detection,
          scalingFactors
        );

        if (result) {
          results.push(result);
          const pixelCount = result.extractedWidth * result.extractedHeight;
          console.log(`✅ Extracted card ${detection.id}: ${result.extractedWidth}x${result.extractedHeight} (${pixelCount.toLocaleString()} pixels)`);
          console.log(`   Method: ${result.metadata.extractionMethod}`);
          console.log(`   High-res: ${result.metadata.isHighResolution}`);
          console.log(`   Padding applied: ${result.metadata.paddingApplied.x}x${result.metadata.paddingApplied.y} pixels`);
        } else {
          console.warn(`❌ Failed to extract card ${detection.id} - result was null`);
        }
      } catch (error) {
        console.error(`❌ Failed to extract card ${detection.id}:`, error);
      }
    }

    console.log(`\n🎉 HIGH-RESOLUTION EXTRACTION COMPLETE: ${results.length}/${detections.length} cards extracted successfully`);
    return results;
  }

  /**
   * Extract a single card from the original image
   */
  private async extractSingleCard(
    originalImageData: ImageData,
    detection: CardDetection,
    scalingFactors: { x: number; y: number }
  ): Promise<HighResExtractionResult | null> {
    // Determine extraction method based on detection data
    if (detection.isRotated && detection.corners && detection.corners.length === 4) {
      return this.extractRotatedCard(originalImageData, detection, scalingFactors);
    } else {
      return this.extractAxisAlignedCard(originalImageData, detection, scalingFactors);
    }
  }

  /**
   * Extract card using oriented bounding box with perspective correction
   */
  private async extractRotatedCard(
    originalImageData: ImageData,
    detection: CardDetection,
    scalingFactors: { x: number; y: number }
  ): Promise<HighResExtractionResult | null> {
    if (!detection.corners || detection.corners.length !== 4) {
      console.warn('Invalid corners for rotated card extraction');
      return null;
    }

    // Transform corners from model input coordinates to original image coordinates
    const originalCorners = detection.corners.map(corner => {
      // Handle both normalized (0-1) and pixel coordinates
      let x = corner.x;
      let y = corner.y;

      if (x <= 1 && y <= 1) {
        // Normalized coordinates - scale to original image size
        x = x * originalImageData.width;
        y = y * originalImageData.height;
      } else {
        // Pixel coordinates from model input - scale to original image
        x = x * scalingFactors.x;
        y = y * scalingFactors.y;
      }

      return { x: Math.round(x), y: Math.round(y) };
    });

    // Calculate card dimensions and orientation
    const edge1 = {
      x: originalCorners[1].x - originalCorners[0].x,
      y: originalCorners[1].y - originalCorners[0].y
    };
    const edge2 = {
      x: originalCorners[3].x - originalCorners[0].x,
      y: originalCorners[3].y - originalCorners[0].y
    };

    const cardWidth = Math.round(Math.sqrt(edge1.x * edge1.x + edge1.y * edge1.y));
    const cardHeight = Math.round(Math.sqrt(edge2.x * edge2.x + edge2.y * edge2.y));

    // Calculate padding for perspective correction
    const basePaddingX = Math.round(cardWidth * this.options.paddingRatio);
    const basePaddingY = Math.round(cardHeight * this.options.paddingRatio);
    
    // Ensure minimum padding of at least 20 pixels for high-res extraction
    const minPadding = 20;
    const paddingX = Math.max(basePaddingX, minPadding);
    const paddingY = Math.max(basePaddingY, minPadding);
    
    console.log(`   📏 Perspective padding: ${paddingX}x${paddingY} pixels (base: ${basePaddingX}x${basePaddingY}, min: ${minPadding})`);

    // Apply padding to final dimensions
    let finalWidth = cardWidth + 2 * paddingX;
    let finalHeight = cardHeight + 2 * paddingY;

    // Ensure minimum card size (after padding)
    if (finalWidth < this.options.minCardSize || finalHeight < this.options.minCardSize) {
      const scaleFactor = Math.max(
        this.options.minCardSize / finalWidth,
        this.options.minCardSize / finalHeight
      );
      finalWidth = Math.round(finalWidth * scaleFactor);
      finalHeight = Math.round(finalHeight * scaleFactor);
    }

    // Calculate rotation angle
    const rotationAngle = Math.atan2(edge1.y, edge1.x) * (180 / Math.PI);

    if (this.options.enablePerspectiveCorrection) {
      // Use perspective correction for high-quality extraction with padding
      const extractedImageData = await this.applyPerspectiveCorrectionWithPadding(
        originalImageData,
        originalCorners,
        cardWidth,
        cardHeight,
        paddingX,
        paddingY
      );

      if (extractedImageData) {
        return {
          imageData: extractedImageData,
          extractedWidth: finalWidth,
          extractedHeight: finalHeight,
          metadata: {
            cardId: detection.id,
            extractionMethod: 'perspective',
            originalSize: { width: originalImageData.width, height: originalImageData.height },
            modelInputSize: this.options.modelInputSize,
            scalingFactors,
            confidence: detection.confidence,
            rotationAngle,
            corners: originalCorners,
            paddingApplied: { x: paddingX, y: paddingY },
            isHighResolution: true
          }
        };
      }
    }

    // Fallback to bounding rectangle extraction
    return this.extractBoundingRectangle(
      originalImageData,
      originalCorners,
      detection,
      scalingFactors,
      'obb'
    );
  }

  /**
   * Extract card using regular bounding box
   */
  private async extractAxisAlignedCard(
    originalImageData: ImageData,
    detection: CardDetection,
    scalingFactors: { x: number; y: number }
  ): Promise<HighResExtractionResult | null> {
    const bbox = detection.boundingBox;
    console.log(`   🔲 Axis-aligned extraction for card ${detection.id}`);
    console.log(`   📦 Original bbox: ${JSON.stringify(bbox)}`);

    // Transform coordinates from model input to original image
    let x = bbox.x;
    let y = bbox.y;
    let width = bbox.width;
    let height = bbox.height;

    console.log(`   📏 Raw coordinates: x=${x}, y=${y}, w=${width}, h=${height}`);

    // Handle both normalized (0-1) and pixel coordinates
    if (x <= 1 && y <= 1 && width <= 1 && height <= 1) {
      // Normalized coordinates - scale to original image size
      console.log(`   🔢 Detected normalized coordinates, scaling to original image size`);
      console.log(`   🔍 COORDINATE DEBUG: Normalized values: x=${x}, y=${y}, w=${width}, h=${height}`);
      console.log(`   🔍 COORDINATE DEBUG: Original image size: ${originalImageData.width}x${originalImageData.height}`);
      x = Math.round(x * originalImageData.width);
      y = Math.round(y * originalImageData.height);
      width = Math.round(width * originalImageData.width);
      height = Math.round(height * originalImageData.height);
      console.log(`   📐 Scaled to original: x=${x}, y=${y}, w=${width}, h=${height}`);
      console.log(`   🔍 COORDINATE DEBUG: Aspect ratio: ${(width/height).toFixed(3)} (${width > height ? 'horizontal' : 'vertical'})`);
    } else {
      // Pixel coordinates from model input - scale to original image
      console.log(`   🔢 Detected pixel coordinates, scaling by factors`);
      console.log(`   🔍 COORDINATE DEBUG: Pixel values: x=${x}, y=${y}, w=${width}, h=${height}`);
      console.log(`   🔍 COORDINATE DEBUG: Scaling factors: x=${scalingFactors.x.toFixed(3)}, y=${scalingFactors.y.toFixed(3)}`);
      const originalX = x, originalY = y, originalW = width, originalH = height;
      x = Math.round(x * scalingFactors.x);
      y = Math.round(y * scalingFactors.y);
      width = Math.round(width * scalingFactors.x);
      height = Math.round(height * scalingFactors.y);
      console.log(`   📐 Scaled from (${originalX}, ${originalY}, ${originalW}, ${originalH}) to (${x}, ${y}, ${width}, ${height})`);
      console.log(`   🔍 COORDINATE DEBUG: Aspect ratio: ${(width/height).toFixed(3)} (${width > height ? 'horizontal' : 'vertical'})`);
    }

    // Apply padding - ensure minimum padding for high-resolution extraction
    const basePaddingX = Math.round(width * this.options.paddingRatio);
    const basePaddingY = Math.round(height * this.options.paddingRatio);
    
    // Ensure minimum padding of at least 20 pixels for high-res extraction
    const minPadding = 20;
    const paddingX = Math.max(basePaddingX, minPadding);
    const paddingY = Math.max(basePaddingY, minPadding);
    
    console.log(`   📏 Calculated padding: ${paddingX}x${paddingY} pixels (base: ${basePaddingX}x${basePaddingY}, min: ${minPadding})`);
    console.log(`   📏 Padding ratio: ${(this.options.paddingRatio * 100).toFixed(1)}% of ${width}x${height}`);

    // Calculate padded coordinates with boundary checks
    const paddedX = Math.max(0, x - paddingX);
    const paddedY = Math.max(0, y - paddingY);
    const paddedWidth = Math.min(
      width + 2 * paddingX,
      originalImageData.width - paddedX
    );
    const paddedHeight = Math.min(
      height + 2 * paddingY,
      originalImageData.height - paddedY
    );

    console.log(`   📦 Final extraction region: (${paddedX}, ${paddedY}) ${paddedWidth}x${paddedHeight}`);
    console.log(`   📊 Size comparison: original ${width}x${height} → padded ${paddedWidth}x${paddedHeight}`);

    // Validate dimensions
    if (paddedWidth < this.options.minCardSize || paddedHeight < this.options.minCardSize) {
      console.warn(`❌ Card too small: ${paddedWidth}x${paddedHeight} < ${this.options.minCardSize}`);
      return null;
    }

    if (paddedWidth > this.options.maxCardSize || paddedHeight > this.options.maxCardSize) {
      console.warn(`❌ Card too large: ${paddedWidth}x${paddedHeight} > ${this.options.maxCardSize}`);
      return null;
    }

    // Extract the card region
    console.log(`   ✂️ Cropping rectangle: (${paddedX}, ${paddedY}) ${paddedWidth}x${paddedHeight}`);
    const extractedImageData = this.cropRectangle(
      originalImageData,
      paddedX,
      paddedY,
      paddedWidth,
      paddedHeight
    );

    if (!extractedImageData) {
      console.warn(`❌ Failed to crop rectangle for card ${detection.id}`);
      return null;
    }

    console.log(`   ✅ Successfully extracted: ${extractedImageData.width}x${extractedImageData.height}`);

    return {
      imageData: extractedImageData,
      extractedWidth: paddedWidth,
      extractedHeight: paddedHeight,
      metadata: {
        cardId: detection.id,
        extractionMethod: 'bbox',
        originalSize: { width: originalImageData.width, height: originalImageData.height },
        modelInputSize: this.options.modelInputSize,
        scalingFactors,
        confidence: detection.confidence,
        paddingApplied: { x: paddingX, y: paddingY },
        isHighResolution: true
      }
    };
  }

  /**
   * Extract bounding rectangle from corner points (fallback for OBB)
   */
  private extractBoundingRectangle(
    originalImageData: ImageData,
    corners: Array<{ x: number; y: number }>,
    detection: CardDetection,
    scalingFactors: { x: number; y: number },
    method: 'obb' | 'bbox'
  ): HighResExtractionResult | null {
    // Calculate bounding rectangle
    const minX = Math.min(...corners.map(c => c.x));
    const maxX = Math.max(...corners.map(c => c.x));
    const minY = Math.min(...corners.map(c => c.y));
    const maxY = Math.max(...corners.map(c => c.y));

    const width = maxX - minX;
    const height = maxY - minY;

    // Apply padding - ensure minimum padding for high-resolution extraction
    const basePaddingX = Math.round(width * this.options.paddingRatio);
    const basePaddingY = Math.round(height * this.options.paddingRatio);
    
    // Ensure minimum padding of at least 20 pixels for high-res extraction
    const minPadding = 20;
    const paddingX = Math.max(basePaddingX, minPadding);
    const paddingY = Math.max(basePaddingY, minPadding);

    const paddedX = Math.max(0, minX - paddingX);
    const paddedY = Math.max(0, minY - paddingY);
    const paddedWidth = Math.min(
      width + 2 * paddingX,
      originalImageData.width - paddedX
    );
    const paddedHeight = Math.min(
      height + 2 * paddingY,
      originalImageData.height - paddedY
    );

    // Extract the region
    const extractedImageData = this.cropRectangle(
      originalImageData,
      paddedX,
      paddedY,
      paddedWidth,
      paddedHeight
    );

    if (!extractedImageData) {
      return null;
    }

    return {
      imageData: extractedImageData,
      extractedWidth: paddedWidth,
      extractedHeight: paddedHeight,
      metadata: {
        cardId: detection.id,
        extractionMethod: method,
        originalSize: { width: originalImageData.width, height: originalImageData.height },
        modelInputSize: this.options.modelInputSize,
        scalingFactors,
        confidence: detection.confidence,
        corners,
        paddingApplied: { x: paddingX, y: paddingY },
        isHighResolution: true
      }
    };
  }

  /**
   * Apply perspective correction to extract a rotated card
   */
  private async applyPerspectiveCorrectionWithPadding(
    originalImageData: ImageData,
    corners: Array<{ x: number; y: number }>,
    cardWidth: number,
    cardHeight: number,
    paddingX: number,
    paddingY: number
  ): Promise<ImageData | null> {
    try {
      // Create source canvas
      const sourceCanvas = document.createElement('canvas');
      const sourceCtx = sourceCanvas.getContext('2d');
      if (!sourceCtx) return null;

      sourceCanvas.width = originalImageData.width;
      sourceCanvas.height = originalImageData.height;
      sourceCtx.putImageData(originalImageData, 0, 0);

      // Create destination canvas with padding
      const destCanvas = document.createElement('canvas');
      const destCtx = destCanvas.getContext('2d');
      if (!destCtx) return null;

      const totalWidth = cardWidth + 2 * paddingX;
      const totalHeight = cardHeight + 2 * paddingY;
      destCanvas.width = totalWidth;
      destCanvas.height = totalHeight;

      console.log(`   📐 Perspective canvas: ${totalWidth}x${totalHeight} (card: ${cardWidth}x${cardHeight}, padding: ${paddingX}x${paddingY})`);

      // Fill with a neutral background color (light gray)
      destCtx.fillStyle = '#f0f0f0';
      destCtx.fillRect(0, 0, totalWidth, totalHeight);

      // Define destination rectangle (card area centered within the padded canvas)
      const destCorners = [
        { x: paddingX, y: paddingY },
        { x: paddingX + cardWidth, y: paddingY },
        { x: paddingX + cardWidth, y: paddingY + cardHeight },
        { x: paddingX, y: paddingY + cardHeight }
      ];

      console.log(`   📍 Destination corners: TL(${paddingX},${paddingY}) TR(${paddingX + cardWidth},${paddingY}) BR(${paddingX + cardWidth},${paddingY + cardHeight}) BL(${paddingX},${paddingY + cardHeight})`);
      console.log(`   📏 Expected padding: Left=${paddingX}, Right=${totalWidth - (paddingX + cardWidth)}, Top=${paddingY}, Bottom=${totalHeight - (paddingY + cardHeight)}`);

      // Apply perspective transformation using transform matrix
      const transform = this.calculatePerspectiveTransform(corners, destCorners);
      if (!transform) {
        console.warn('Failed to calculate perspective transform');
        return null;
      }

      // Apply the transformation
      destCtx.save();
      destCtx.setTransform(
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f
      );

      // Draw the transformed image
      destCtx.drawImage(sourceCanvas, 0, 0);
      destCtx.restore();

      // Get the result as ImageData
      return destCtx.getImageData(0, 0, totalWidth, totalHeight);

    } catch (error) {
      console.error('Perspective correction with padding failed:', error);
      return null;
    }
  }

  private async applyPerspectiveCorrection(
    originalImageData: ImageData,
    corners: Array<{ x: number; y: number }>,
    targetWidth: number,
    targetHeight: number
  ): Promise<ImageData | null> {
    try {
      // Create source canvas
      const sourceCanvas = document.createElement('canvas');
      const sourceCtx = sourceCanvas.getContext('2d');
      if (!sourceCtx) return null;

      sourceCanvas.width = originalImageData.width;
      sourceCanvas.height = originalImageData.height;
      sourceCtx.putImageData(originalImageData, 0, 0);

      // Create destination canvas
      const destCanvas = document.createElement('canvas');
      const destCtx = destCanvas.getContext('2d');
      if (!destCtx) return null;

      destCanvas.width = targetWidth;
      destCanvas.height = targetHeight;

      // Define destination rectangle (straightened card)
      const destCorners = [
        { x: 0, y: 0 },
        { x: targetWidth, y: 0 },
        { x: targetWidth, y: targetHeight },
        { x: 0, y: targetHeight }
      ];

      // Apply perspective transformation using transform matrix
      const transform = this.calculatePerspectiveTransform(corners, destCorners);
      if (!transform) {
        console.warn('Failed to calculate perspective transform');
        return null;
      }

      // Apply the transformation
      destCtx.save();
      destCtx.setTransform(
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f
      );

      // Draw the transformed image
      destCtx.drawImage(sourceCanvas, 0, 0);
      destCtx.restore();

      // Get the result
      return destCtx.getImageData(0, 0, targetWidth, targetHeight);

    } catch (error) {
      console.warn('Perspective correction failed:', error);
      return null;
    }
  }

  /**
   * Calculate perspective transformation matrix (simplified 2D transform)
   */
  private calculatePerspectiveTransform(
    sourceCorners: Array<{ x: number; y: number }>,
    destCorners: Array<{ x: number; y: number }>
  ): DOMMatrix | null {
    try {
      // For now, use a simple affine transformation
      // This is a simplified approach - full perspective correction would require
      // more complex matrix calculations

      const src = sourceCorners;
      const dst = destCorners;

      // Calculate transformation based on first three points
      const srcX1 = src[0].x, srcY1 = src[0].y;
      const srcX2 = src[1].x, srcY2 = src[1].y;
      const srcX3 = src[3].x, srcY3 = src[3].y;

      const dstX1 = dst[0].x, dstY1 = dst[0].y;
      const dstX2 = dst[1].x, dstY2 = dst[1].y;
      const dstX3 = dst[3].x, dstY3 = dst[3].y;

      // Calculate affine transformation matrix
      const matrix = new DOMMatrix();

      // This is a simplified transformation - for production use,
      // consider using a proper perspective transformation library
      const scaleX = (dstX2 - dstX1) / (srcX2 - srcX1);
      const scaleY = (dstY3 - dstY1) / (srcY3 - srcY1);

      // Apply transformations in correct order: translate source to origin, scale, then translate to destination
      matrix.translateSelf(dstX1, dstY1);  // Move to destination position
      matrix.scaleSelf(scaleX, scaleY);    // Scale to match destination size
      matrix.translateSelf(-srcX1, -srcY1); // Move source corner to origin

      console.log(`   🔄 Transform: src(${srcX1},${srcY1}) → dst(${dstX1},${dstY1}), scale(${scaleX.toFixed(3)},${scaleY.toFixed(3)})`);

      return matrix;

    } catch (error) {
      console.warn('Failed to calculate perspective transform:', error);
      return null;
    }
  }

  /**
   * Crop a rectangle from the source image
   */
  private cropRectangle(
    sourceImageData: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
  ): ImageData | null {
    try {
      console.log(`   🔍 cropRectangle DEBUG: Input coordinates: (${x}, ${y}) ${width}x${height}`);
      console.log(`   🔍 cropRectangle DEBUG: Source image: ${sourceImageData.width}x${sourceImageData.height}`);
      
      // Store original values for debugging
      const originalX = x, originalY = y, originalWidth = width, originalHeight = height;
      
      // Ensure coordinates are within bounds
      x = Math.max(0, Math.min(x, sourceImageData.width - 1));
      y = Math.max(0, Math.min(y, sourceImageData.height - 1));
      width = Math.min(width, sourceImageData.width - x);
      height = Math.min(height, sourceImageData.height - y);

      console.log(`   🔍 cropRectangle DEBUG: Bounded coordinates: (${x}, ${y}) ${width}x${height}`);
      console.log(`   🔍 cropRectangle DEBUG: Coordinate changes: x(${originalX}→${x}), y(${originalY}→${y}), w(${originalWidth}→${width}), h(${originalHeight}→${height})`);

      if (width <= 0 || height <= 0) {
        console.error(`   ❌ cropRectangle DEBUG: Invalid dimensions after bounding: ${width}x${height}`);
        return null;
      }

      // Create canvas for cropping
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.error('   ❌ cropRectangle DEBUG: Failed to get destination canvas context');
        return null;
      }

      canvas.width = width;
      canvas.height = height;
      console.log(`   🔍 cropRectangle DEBUG: Created destination canvas: ${canvas.width}x${canvas.height}`);

      // Create source canvas
      const sourceCanvas = document.createElement('canvas');
      const sourceCtx = sourceCanvas.getContext('2d');
      if (!sourceCtx) {
        console.error('   ❌ cropRectangle DEBUG: Failed to get source canvas context');
        return null;
      }

      sourceCanvas.width = sourceImageData.width;
      sourceCanvas.height = sourceImageData.height;
      console.log(`   🔍 cropRectangle DEBUG: Created source canvas: ${sourceCanvas.width}x${sourceCanvas.height}`);
      
      // Put image data on source canvas
      sourceCtx.putImageData(sourceImageData, 0, 0);
      console.log(`   🔍 cropRectangle DEBUG: Put ImageData on source canvas`);

      // Draw the cropped region
      console.log(`   🔍 cropRectangle DEBUG: Drawing region: src(${x}, ${y}, ${width}, ${height}) → dest(0, 0, ${width}, ${height})`);
      ctx.drawImage(
        sourceCanvas,
        x, y, width, height,  // Source rectangle
        0, 0, width, height   // Destination rectangle
      );

      // Get the result
      const result = ctx.getImageData(0, 0, width, height);
      console.log(`   🔍 cropRectangle DEBUG: Extracted ImageData: ${result.width}x${result.height}, data length: ${result.data.length}`);
      
      // Check if the result is actually blank (all pixels are transparent/black)
      let nonZeroPixels = 0;
      for (let i = 0; i < result.data.length; i += 4) {
        if (result.data[i] !== 0 || result.data[i + 1] !== 0 || result.data[i + 2] !== 0 || result.data[i + 3] !== 0) {
          nonZeroPixels++;
        }
      }
      console.log(`   🔍 cropRectangle DEBUG: Non-zero pixels: ${nonZeroPixels}/${result.data.length / 4} (${((nonZeroPixels / (result.data.length / 4)) * 100).toFixed(1)}%)`);
      
      if (nonZeroPixels === 0) {
        console.warn(`   ⚠️ cropRectangle DEBUG: Extracted image appears to be completely blank!`);
      }

      return result;

    } catch (error) {
      console.error('❌ cropRectangle DEBUG: Exception occurred:', error);
      return null;
    }
  }

  /**
   * Get ImageData from various image sources
   */
  private async getImageData(
    image: HTMLImageElement | HTMLCanvasElement | ImageData,
    originalSize?: { width: number; height: number }
  ): Promise<ImageData | null> {
    try {
      if (image instanceof ImageData) {
        return image;
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      if (image instanceof HTMLImageElement) {
        // Use original size if provided, otherwise use natural size
        canvas.width = originalSize?.width || image.naturalWidth;
        canvas.height = originalSize?.height || image.naturalHeight;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      } else if (image instanceof HTMLCanvasElement) {
        canvas.width = originalSize?.width || image.width;
        canvas.height = originalSize?.height || image.height;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      }

      return ctx.getImageData(0, 0, canvas.width, canvas.height);

    } catch (error) {
      console.error('Failed to get image data:', error);
      return null;
    }
  }

  /**
   * Convert ImageData to blob for saving
   */
  static async imageDataToBlob(imageData: ImageData, format: string = 'image/jpeg', quality: number = 0.9): Promise<Blob> {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);

    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert canvas to blob'));
        }
      }, format, quality);
    });
  }

  /**
   * Save extracted card as downloadable file
   */
  static async saveCard(
    result: HighResExtractionResult,
    filename?: string,
    format: string = 'image/jpeg',
    quality: number = 0.9
  ): Promise<void> {
    const blob = await this.imageDataToBlob(result.imageData, format, quality);
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `card_${result.metadata.cardId}_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  }
}
