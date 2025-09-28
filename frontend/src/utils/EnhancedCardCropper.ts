import { CardDetection } from '../store/slices/inferenceSlice';
import { HighResCardExtractor, HighResExtractionResult, HighResExtractionOptions } from './HighResCardExtractor';
import { CardCropper, CropResult } from './CardCropper';

export interface EnhancedCropResult extends CropResult {
  // Add aliases for consistency with HighResExtractionResult
  extractedWidth: number;
  extractedHeight: number;
  metadata: {
    cardId: string;
    extractionMethod: 'bbox' | 'obb' | 'perspective' | 'fallback';
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
 * Enhanced Card Cropper that supports high-resolution extraction
 * 
 * This class extends the existing CardCropper functionality to support extracting
 * cards from original full-resolution images while using detection coordinates
 * from models that were run on downsampled inputs.
 */
export class EnhancedCardCropper {
  private highResExtractor: HighResCardExtractor;
  
  constructor(options: Partial<HighResExtractionOptions> = {}) {
    this.highResExtractor = new HighResCardExtractor(options);
  }

  /**
   * Extract cards with high-resolution support
   * 
   * @param sourceImageData - The downsampled image data used for inference
   * @param detections - Detection results from the model
   * @param originalImage - Optional original full-resolution image
   * @param originalImageSize - Size of the original image if not provided directly
   * @returns Array of enhanced crop results
   */
  async extractCards(
    sourceImageData: ImageData,
    detections: CardDetection[],
    originalImage?: HTMLImageElement | HTMLCanvasElement | ImageData,
    originalImageSize?: { width: number; height: number }
  ): Promise<EnhancedCropResult[]> {
    
    console.log(`🔍 ENHANCED CARD EXTRACTION STARTING`);
    console.log(`📊 Detections: ${detections.length}`);
    console.log(`📐 Source image: ${sourceImageData.width}x${sourceImageData.height}`);
    console.log(`🖼️ Original image provided: ${!!originalImage}`);
    console.log(`📏 Original size provided: ${!!originalImageSize}`);
    
    if (originalImageSize) {
      console.log(`📏 Original size: ${originalImageSize.width}x${originalImageSize.height}`);
    }
    
    if (originalImage || originalImageSize) {
      console.log(`🎯 HIGH-RESOLUTION MODE ENABLED - Using HighResCardExtractor`);
      return this.extractHighResolution(sourceImageData, detections, originalImage, originalImageSize);
    } else {
      console.log(`📱 STANDARD RESOLUTION MODE (FALLBACK) - Using CardCropper`);
      return this.extractStandardResolution(sourceImageData, detections);
    }
  }

  /**
   * Extract cards at high resolution from original image
   */
  private async extractHighResolution(
    sourceImageData: ImageData,
    detections: CardDetection[],
    originalImage?: HTMLImageElement | HTMLCanvasElement | ImageData,
    originalImageSize?: { width: number; height: number }
  ): Promise<EnhancedCropResult[]> {
    
    console.log(`🎯 HIGH-RESOLUTION EXTRACTION STARTING`);
    console.log(`📐 Source: ${sourceImageData.width}x${sourceImageData.height}`);
    console.log(`🖼️ Original image: ${!!originalImage}`);
    console.log(`📏 Original size: ${originalImageSize ? `${originalImageSize.width}x${originalImageSize.height}` : 'none'}`);
    
    try {
      // If we have the original image, use it directly
      if (originalImage) {
        console.log(`✅ Using original image directly for high-res extraction`);
        const highResResults = await this.highResExtractor.extractCards(
          originalImage,
          detections,
          originalImageSize
        );
        
        console.log(`✅ High-res extractor returned ${highResResults.length} results`);
        return this.convertHighResToEnhanced(highResResults);
      }
      
      // If we only have size info, try to reconstruct or upscale
      if (originalImageSize) {
        console.log(`📏 Original size provided: ${originalImageSize.width}x${originalImageSize.height}`);
        
        // Check if we need to upscale significantly
        const scaleX = originalImageSize.width / sourceImageData.width;
        const scaleY = originalImageSize.height / sourceImageData.height;
        
        console.log(`📊 Scale factors: x=${scaleX.toFixed(2)}, y=${scaleY.toFixed(2)}`);
        
        if (scaleX > 1.5 || scaleY > 1.5) {
          console.log(`🔍 Significant upscaling needed: ${scaleX.toFixed(2)}x, ${scaleY.toFixed(2)}x`);
          
          // Create upscaled version of source image
          const upscaledImage = await this.upscaleImage(sourceImageData, originalImageSize);
          if (upscaledImage) {
            console.log(`✅ Created upscaled image: ${upscaledImage.width}x${upscaledImage.height}`);
            const highResResults = await this.highResExtractor.extractCards(
              upscaledImage,
              detections,
              originalImageSize
            );
            
            console.log(`✅ High-res extractor (upscaled) returned ${highResResults.length} results`);
            return this.convertHighResToEnhanced(highResResults);
          } else {
            console.warn(`❌ Failed to create upscaled image`);
          }
        } else {
          console.log(`📊 Scale factors too small (${scaleX.toFixed(2)}x, ${scaleY.toFixed(2)}x), not worth upscaling`);
        }
      }
      
      // Fallback to standard extraction
      console.log(`⚠️ HIGH-RESOLUTION FAILED - Falling back to standard resolution extraction`);
      return this.extractStandardResolution(sourceImageData, detections);
      
    } catch (error) {
      console.error('❌ High-resolution extraction failed:', error);
      console.log(`⚠️ EXCEPTION - Falling back to standard resolution extraction`);
      return this.extractStandardResolution(sourceImageData, detections);
    }
  }

  /**
   * Extract cards using standard resolution (existing CardCropper logic)
   */
  private async extractStandardResolution(
    sourceImageData: ImageData,
    detections: CardDetection[]
  ): Promise<EnhancedCropResult[]> {
    
    console.log('📱 STANDARD RESOLUTION EXTRACTION (FALLBACK)');
    console.log(`📐 Source image: ${sourceImageData.width}x${sourceImageData.height}`);
    console.log(`🎴 Processing ${detections.length} detections with CardCropper`);
    
    // Use existing CardCropper for standard extraction
    const standardResults = CardCropper.extractCards(sourceImageData, detections);
    
    console.log(`📱 CardCropper returned ${standardResults.length} results`);
    standardResults.forEach((result, index) => {
      console.log(`   Card ${index + 1}: ${result.croppedWidth}x${result.croppedHeight} pixels`);
    });
    
    // Convert to enhanced format
    const enhancedResults = standardResults.map((result, index) => {
      const detection = detections[index];
      
      // Calculate what the padding should have been for CardCropper
      const bbox = detection.boundingBox;
      let width = bbox.width;
      let height = bbox.height;
      
      // Handle normalized coordinates
      if (width <= 1 && height <= 1) {
        width = Math.round(width * sourceImageData.width);
        height = Math.round(height * sourceImageData.height);
      }
      
      const expectedPaddingX = Math.round(width * 0.1); // CardCropper uses 10% padding
      const expectedPaddingY = Math.round(height * 0.1);
      
      console.log(`   📱 Fallback card ${index + 1}: ${result.croppedWidth}x${result.croppedHeight}`);
      console.log(`   📱 Expected padding: ${expectedPaddingX}x${expectedPaddingY} (10% of ${width}x${height})`);
      console.log(`   📱 Original bbox: ${JSON.stringify(bbox)}`);
      
      return {
        ...result,
        extractedWidth: result.croppedWidth,
        extractedHeight: result.croppedHeight,
        metadata: {
          cardId: detection.id,
          extractionMethod: 'fallback' as const,
          originalSize: { width: sourceImageData.width, height: sourceImageData.height },
          modelInputSize: { width: sourceImageData.width, height: sourceImageData.height },
          scalingFactors: { x: 1, y: 1 },
          confidence: detection.confidence,
          corners: detection.corners,
          paddingApplied: { x: expectedPaddingX, y: expectedPaddingY }, // Show what CardCropper should have applied
          isHighResolution: false
        }
      };
    });
    
    console.log('📱 Standard resolution extraction complete');
    return enhancedResults;
  }

  /**
   * Convert high-resolution results to enhanced crop results
   */
  private convertHighResToEnhanced(highResResults: HighResExtractionResult[]): EnhancedCropResult[] {
    return highResResults.map(result => ({
      imageData: result.imageData,
      croppedWidth: result.extractedWidth,
      croppedHeight: result.extractedHeight,
      extractedWidth: result.extractedWidth,
      extractedHeight: result.extractedHeight,
      metadata: {
        ...result.metadata,
        isHighResolution: true
      }
    }));
  }

  /**
   * Upscale image using canvas interpolation
   */
  private async upscaleImage(
    sourceImageData: ImageData,
    targetSize: { width: number; height: number }
  ): Promise<HTMLCanvasElement | null> {
    try {
      // Create source canvas
      const sourceCanvas = document.createElement('canvas');
      const sourceCtx = sourceCanvas.getContext('2d');
      if (!sourceCtx) return null;

      sourceCanvas.width = sourceImageData.width;
      sourceCanvas.height = sourceImageData.height;
      sourceCtx.putImageData(sourceImageData, 0, 0);

      // Create target canvas
      const targetCanvas = document.createElement('canvas');
      const targetCtx = targetCanvas.getContext('2d');
      if (!targetCtx) return null;

      targetCanvas.width = targetSize.width;
      targetCanvas.height = targetSize.height;

      // Use high-quality scaling
      targetCtx.imageSmoothingEnabled = true;
      targetCtx.imageSmoothingQuality = 'high';

      // Draw upscaled image
      targetCtx.drawImage(
        sourceCanvas,
        0, 0, sourceImageData.width, sourceImageData.height,
        0, 0, targetSize.width, targetSize.height
      );

      return targetCanvas;

    } catch (error) {
      console.error('Failed to upscale image:', error);
      return null;
    }
  }

  /**
   * Extract cards from camera stream with original image support
   * 
   * This method is designed to work with the camera interface where we have
   * access to both the processed frame and potentially the original camera stream
   */
  static async extractFromCameraFrame(
    processedFrame: ImageData,
    detections: CardDetection[],
    videoElement?: HTMLVideoElement,
    options: Partial<HighResExtractionOptions> = {}
  ): Promise<EnhancedCropResult[]> {
    
    console.log('🎯 EnhancedCardCropper.extractFromCameraFrame called');
    console.log(`📐 Processed frame: ${processedFrame.width}x${processedFrame.height}`);
    console.log(`🎥 Video element provided: ${!!videoElement}`);
    
    if (videoElement) {
      console.log(`📹 Video element dimensions: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
      console.log(`📹 Video element ready state: ${videoElement.readyState}`);
      console.log(`📹 Video element current time: ${videoElement.currentTime}`);
    }
    
    const enhancedCropper = new EnhancedCardCropper(options);
    
    // If we have access to the video element, capture at native resolution
    if (videoElement && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
      console.log(`✅ Using video element for high-res extraction: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
      
      // Capture frame from video at native resolution
      const nativeFrame = await EnhancedCardCropper.captureVideoFrame(videoElement);
      if (nativeFrame) {
        console.log(`✅ Captured native frame: ${nativeFrame.width}x${nativeFrame.height}`);
        
        const results = await enhancedCropper.extractCards(
          processedFrame,
          detections,
          nativeFrame,
          { width: videoElement.videoWidth, height: videoElement.videoHeight }
        );
        
        console.log(`✅ High-res extraction complete: ${results.length} cards extracted`);
        results.forEach((result, index) => {
          console.log(`   Card ${index + 1}: ${result.extractedWidth}x${result.extractedHeight} (${result.metadata.extractionMethod}, high-res: ${result.metadata.isHighResolution})`);
        });
        
        return results;
      } else {
        console.warn('❌ Failed to capture native frame from video element');
      }
    } else {
      console.warn('⚠️ Video element not available or has no dimensions, falling back to processed frame');
    }
    
    // Fallback to processed frame
    console.log('📱 Falling back to standard resolution extraction');
    const results = await enhancedCropper.extractCards(processedFrame, detections);
    
    console.log(`📱 Standard extraction complete: ${results.length} cards extracted`);
    results.forEach((result, index) => {
      console.log(`   Card ${index + 1}: ${result.extractedWidth}x${result.extractedHeight} (${result.metadata.extractionMethod}, high-res: ${result.metadata.isHighResolution})`);
    });
    
    return results;
  }

  /**
   * Capture a frame from video element at native resolution
   */
  static async captureVideoFrame(videoElement: HTMLVideoElement): Promise<HTMLCanvasElement | null> {
    try {
      console.log('📹 Capturing video frame...');
      console.log(`   Video dimensions: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
      console.log(`   Video ready state: ${videoElement.readyState}`);
      console.log(`   Video current time: ${videoElement.currentTime}`);
      console.log(`   Video paused: ${videoElement.paused}`);
      
      if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
        console.warn('❌ Video element has no dimensions');
        return null;
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.warn('❌ Failed to get canvas context');
        return null;
      }

      // Set canvas to video's native resolution
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      console.log(`   Created canvas: ${canvas.width}x${canvas.height}`);

      // Draw current video frame
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      console.log('✅ Successfully captured video frame to canvas');

      return canvas;

    } catch (error) {
      console.error('❌ Failed to capture video frame:', error);
      return null;
    }
  }

  /**
   * Validate if a detection represents a valid card (enhanced version)
   */
  static isValidCardDetection(detection: CardDetection, minConfidence: number = 0.3): boolean {
    // Use existing validation as base
    if (!CardCropper.isValidCardDetection(detection)) {
      return false;
    }

    // Additional validation for high-resolution extraction
    if (detection.confidence < minConfidence) {
      return false;
    }

    // Validate corners if present
    if (detection.corners && detection.corners.length === 4) {
      // Check if corners form a reasonable quadrilateral
      const corners = detection.corners;
      
      // Calculate area using shoelace formula
      let area = 0;
      for (let i = 0; i < 4; i++) {
        const j = (i + 1) % 4;
        area += corners[i].x * corners[j].y;
        area -= corners[j].x * corners[i].y;
      }
      area = Math.abs(area) / 2;
      
      // Check if area is reasonable (not too small or degenerate)
      if (area < 100) { // Minimum area in pixels
        console.warn(`Card area too small: ${area}`);
        return false;
      }
    }

    return true;
  }

  /**
   * Get extraction quality metrics
   */
  static getExtractionQuality(result: EnhancedCropResult): {
    score: number;
    factors: string[];
  } {
    const factors: string[] = [];
    let score = 0;

    // Base score from confidence
    score += result.metadata.confidence * 40;
    factors.push(`Confidence: ${(result.metadata.confidence * 100).toFixed(1)}%`);

    // Resolution bonus
    if (result.metadata.isHighResolution) {
      score += 20;
      factors.push('High resolution');
    }

    // Size bonus
    const area = result.croppedWidth * result.croppedHeight;
    if (area > 100000) { // > 316x316
      score += 20;
      factors.push('Large size');
    } else if (area > 40000) { // > 200x200
      score += 10;
      factors.push('Medium size');
    }

    // Method bonus
    if (result.metadata.extractionMethod === 'perspective') {
      score += 15;
      factors.push('Perspective corrected');
    } else if (result.metadata.extractionMethod === 'obb') {
      score += 10;
      factors.push('Oriented bounding box');
    }

    // Aspect ratio check
    const aspectRatio = result.croppedWidth / result.croppedHeight;
    if (aspectRatio >= 0.6 && aspectRatio <= 1.6) { // Reasonable card aspect ratio
      score += 5;
      factors.push('Good aspect ratio');
    }

    return {
      score: Math.min(100, Math.max(0, score)),
      factors
    };
  }
}
