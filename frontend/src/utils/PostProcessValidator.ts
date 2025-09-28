import { BackboneModelManager } from './BackboneModelManager';
import { CardDetection } from '../store/slices/inferenceSlice';
import { EnhancedCropResult } from './EnhancedCardCropper';

// Legacy ModelPrediction interface for compatibility
interface ModelPrediction {
  boxes: Float32Array;
  rotatedBoxes?: Float32Array;
  scores: Float32Array;
  classes: Int32Array;
  validDetections: number;
}

// Adapter interface to support BackboneModelManager
interface ModelManagerAdapter {
  predict(imageData: ImageData): Promise<ModelPrediction>;
}

// Adapter class to convert BackboneModelManager to legacy ModelManager interface
class BackboneModelAdapter implements ModelManagerAdapter {
  constructor(private backboneManager: BackboneModelManager) {}

  async predict(imageData: ImageData): Promise<ModelPrediction> {
    const backbonePrediction = await this.backboneManager.predict(imageData);
    
    // Convert BackboneModelPrediction to ModelPrediction format
    const validDetections = backbonePrediction.detections.length;
    const boxes = new Float32Array(validDetections * 4);
    const scores = new Float32Array(validDetections);
    const classes = new Int32Array(validDetections);
    const rotatedBoxes = new Float32Array(validDetections * 8);

    backbonePrediction.detections.forEach((detection, i) => {
      // Convert normalized OBB to axis-aligned bounding box
      const boxIndex = i * 4;
      boxes[boxIndex] = detection.boundingBox.x;
      boxes[boxIndex + 1] = detection.boundingBox.y;
      boxes[boxIndex + 2] = detection.boundingBox.width;
      boxes[boxIndex + 3] = detection.boundingBox.height;
      
      scores[i] = detection.confidence;
      classes[i] = 0; // All detections are cards (class 0)
      
      // Store rotated corners
      const rotatedIndex = i * 8;
      rotatedBoxes[rotatedIndex] = detection.corners.x1;
      rotatedBoxes[rotatedIndex + 1] = detection.corners.y1;
      rotatedBoxes[rotatedIndex + 2] = detection.corners.x2;
      rotatedBoxes[rotatedIndex + 3] = detection.corners.y2;
      rotatedBoxes[rotatedIndex + 4] = detection.corners.x3;
      rotatedBoxes[rotatedIndex + 5] = detection.corners.y3;
      rotatedBoxes[rotatedIndex + 6] = detection.corners.x4;
      rotatedBoxes[rotatedIndex + 7] = detection.corners.y4;
    });

    return {
      boxes,
      rotatedBoxes: validDetections > 0 ? rotatedBoxes : undefined,
      scores,
      classes,
      validDetections
    };
  }
}

export interface PostProcessValidationResult {
  isValid: boolean;
  originalConfidence: number;
  adjustedConfidence: number;
  confidenceAdjustment: number;
  validationScore: number;
  validationMetrics: {
    detectionFound: boolean;
    detectionConfidence: number;
    detectionCount: number;
    aspectRatioMatch: boolean;
    sizeConsistency: boolean;
    cardLikeFeatures: boolean;
  };
  recommendations: string[];
}

export interface PostProcessValidationOptions {
  confidenceThreshold: number;
  maxConfidenceBoost: number;
  maxConfidencePenalty: number;
  enableAspectRatioValidation: boolean;
  enableSizeValidation: boolean;
  enableFeatureValidation: boolean;
}

/**
 * PostProcessValidator
 * 
 * Validates extracted card images by running them through the model again
 * to ensure the model still detects a card in the extracted region.
 * Adjusts confidence scores based on validation results.
 * 
 * Updated to work with BackboneModelManager through adapter pattern.
 */
export class PostProcessValidator {
  private modelAdapter: ModelManagerAdapter;
  private options: PostProcessValidationOptions;

  constructor(
    backboneModelManager: BackboneModelManager,
    options: Partial<PostProcessValidationOptions> = {}
  ) {
    this.modelAdapter = new BackboneModelAdapter(backboneModelManager);
    this.options = {
      confidenceThreshold: 0.25, // Lower threshold for validation
      maxConfidenceBoost: 0.15,  // Max 15% boost for valid detections
      maxConfidencePenalty: 0.30, // Max 30% penalty for invalid detections
      enableAspectRatioValidation: true,
      enableSizeValidation: true,
      enableFeatureValidation: true,
      ...options
    };
  }

  /**
   * Create PostProcessValidator from BackboneModelManager
   */
  static fromBackboneManager(
    backboneManager: BackboneModelManager,
    options: Partial<PostProcessValidationOptions> = {}
  ): PostProcessValidator {
    return new PostProcessValidator(backboneManager, options);
  }

  /**
   * Validate a single extracted card by running inference on it
   */
  async validateExtractedCard(
    extractedCard: EnhancedCropResult,
    originalDetection: CardDetection
  ): Promise<PostProcessValidationResult> {
    console.log(`🔍 PostProcess validation for card ${extractedCard.metadata.cardId}`);
    console.log(`   Original confidence: ${(originalDetection.confidence * 100).toFixed(1)}%`);
    console.log(`   Extracted size: ${extractedCard.extractedWidth}x${extractedCard.extractedHeight}`);

    const validationMetrics = {
      detectionFound: false,
      detectionConfidence: 0,
      detectionCount: 0,
      aspectRatioMatch: false,
      sizeConsistency: false,
      cardLikeFeatures: false
    };

    const recommendations: string[] = [];
    let validationScore = 0;

    try {
      // Run inference on the extracted card image using backbone model
      console.log(`   🤖 Running backbone inference on extracted card...`);
      const prediction = await this.modelAdapter.predict(extractedCard.imageData);
      
      validationMetrics.detectionCount = prediction.validDetections;
      console.log(`   📊 Validation inference found ${prediction.validDetections} detections`);

      // Check if model still detects a card
      if (prediction.validDetections > 0) {
        validationMetrics.detectionFound = true;
        
        // Find the highest confidence detection
        let maxConfidence = 0;
        for (let i = 0; i < prediction.validDetections; i++) {
          if (prediction.scores[i] > maxConfidence) {
            maxConfidence = prediction.scores[i];
          }
        }
        
        validationMetrics.detectionConfidence = maxConfidence;
        console.log(`   ✅ Card detected with confidence: ${(maxConfidence * 100).toFixed(1)}%`);
        
        // Validation score based on detection confidence
        if (maxConfidence >= this.options.confidenceThreshold) {
          validationScore += 40; // Strong detection
          console.log(`   ✅ Detection above threshold (${(this.options.confidenceThreshold * 100).toFixed(0)}%)`);
        } else {
          validationScore += 20; // Weak detection
          console.log(`   ⚠️ Detection below threshold but still found`);
          recommendations.push('Detection confidence is low - consider adjusting extraction parameters');
        }
      } else {
        console.log(`   ❌ No card detected in extracted image`);
        recommendations.push('No card detected in extracted region - extraction may be inaccurate');
      }

      // Aspect ratio validation
      if (this.options.enableAspectRatioValidation) {
        const aspectRatio = extractedCard.extractedWidth / extractedCard.extractedHeight;
        const isValidAspectRatio = aspectRatio >= 0.5 && aspectRatio <= 2.0; // Reasonable card range
        validationMetrics.aspectRatioMatch = isValidAspectRatio;
        
        if (isValidAspectRatio) {
          validationScore += 15;
          console.log(`   ✅ Valid aspect ratio: ${aspectRatio.toFixed(2)}`);
        } else {
          console.log(`   ❌ Invalid aspect ratio: ${aspectRatio.toFixed(2)}`);
          recommendations.push(`Unusual aspect ratio (${aspectRatio.toFixed(2)}) - may not be a complete card`);
        }
      }

      // Size validation
      if (this.options.enableSizeValidation) {
        const area = extractedCard.extractedWidth * extractedCard.extractedHeight;
        const isReasonableSize = area >= 10000 && area <= 2000000; // 100x100 to 1414x1414
        validationMetrics.sizeConsistency = isReasonableSize;
        
        if (isReasonableSize) {
          validationScore += 15;
          console.log(`   ✅ Reasonable size: ${area.toLocaleString()} pixels`);
        } else {
          console.log(`   ❌ Unreasonable size: ${area.toLocaleString()} pixels`);
          recommendations.push(`Unusual card size (${area.toLocaleString()} pixels) - may be over/under-extracted`);
        }
      }

      // Feature validation (basic image analysis)
      if (this.options.enableFeatureValidation) {
        const hasCardFeatures = await this.analyzeCardFeatures(extractedCard.imageData);
        validationMetrics.cardLikeFeatures = hasCardFeatures;
        
        if (hasCardFeatures) {
          validationScore += 10;
          console.log(`   ✅ Card-like features detected`);
        } else {
          console.log(`   ❌ No clear card features detected`);
          recommendations.push('Image lacks typical card features - may contain background or artifacts');
        }
      }

      // Calculate confidence adjustment
      const confidenceAdjustment = this.calculateConfidenceAdjustment(
        originalDetection.confidence,
        validationScore,
        validationMetrics
      );

      const adjustedConfidence = Math.max(0, Math.min(1, 
        originalDetection.confidence + confidenceAdjustment
      ));

      const isValid = validationScore >= 50; // Require at least 50/100 validation score

      console.log(`   📊 Validation complete:`);
      console.log(`      Score: ${validationScore}/100`);
      console.log(`      Valid: ${isValid}`);
      console.log(`      Confidence: ${(originalDetection.confidence * 100).toFixed(1)}% → ${(adjustedConfidence * 100).toFixed(1)}% (${confidenceAdjustment >= 0 ? '+' : ''}${(confidenceAdjustment * 100).toFixed(1)}%)`);

      return {
        isValid,
        originalConfidence: originalDetection.confidence,
        adjustedConfidence,
        confidenceAdjustment,
        validationScore,
        validationMetrics,
        recommendations
      };

    } catch (error) {
      console.error(`❌ PostProcess validation failed for card ${extractedCard.metadata.cardId}:`, error);
      
      // Return conservative result on error
      return {
        isValid: false,
        originalConfidence: originalDetection.confidence,
        adjustedConfidence: Math.max(0, originalDetection.confidence - 0.1), // Small penalty for validation failure
        confidenceAdjustment: -0.1,
        validationScore: 0,
        validationMetrics,
        recommendations: ['Validation failed due to technical error - confidence reduced as precaution']
      };
    }
  }

  /**
   * Calculate confidence adjustment based on validation results
   */
  private calculateConfidenceAdjustment(
    originalConfidence: number,
    validationScore: number,
    metrics: PostProcessValidationResult['validationMetrics']
  ): number {
    let adjustment = 0;

    // Base adjustment from validation score
    if (validationScore >= 80) {
      // Excellent validation - significant boost
      adjustment = this.options.maxConfidenceBoost * 0.8;
    } else if (validationScore >= 60) {
      // Good validation - moderate boost
      adjustment = this.options.maxConfidenceBoost * 0.5;
    } else if (validationScore >= 40) {
      // Fair validation - small boost
      adjustment = this.options.maxConfidenceBoost * 0.2;
    } else if (validationScore >= 20) {
      // Poor validation - small penalty
      adjustment = -this.options.maxConfidencePenalty * 0.3;
    } else {
      // Very poor validation - significant penalty
      adjustment = -this.options.maxConfidencePenalty * 0.7;
    }

    // Additional adjustments based on specific metrics
    if (metrics.detectionFound) {
      if (metrics.detectionConfidence >= 0.7) {
        adjustment += 0.05; // High confidence detection bonus
      } else if (metrics.detectionConfidence < 0.3) {
        adjustment -= 0.05; // Low confidence detection penalty
      }
    } else {
      adjustment -= 0.1; // No detection found penalty
    }

    // Multiple detections might indicate confusion
    if (metrics.detectionCount > 1) {
      adjustment -= 0.02; // Small penalty for multiple detections
    }

    // Ensure adjustment stays within bounds
    adjustment = Math.max(-this.options.maxConfidencePenalty, 
                        Math.min(this.options.maxConfidenceBoost, adjustment));

    return adjustment;
  }

  /**
   * Analyze image for card-like features
   */
  private async analyzeCardFeatures(imageData: ImageData): Promise<boolean> {
    try {
      // Basic image analysis for card-like features
      const { data, width, height } = imageData;
      
      // Check for reasonable color distribution (not all one color)
      const colorVariance = this.calculateColorVariance(data);
      const hasColorVariation = colorVariance > 100; // Threshold for color variation
      
      // Check for edge content (not mostly empty/background)
      const edgeContent = this.calculateEdgeContent(data, width, height);
      const hasEdgeContent = edgeContent > 0.1; // At least 10% edge content
      
      // Check for reasonable brightness distribution
      const brightnessDistribution = this.calculateBrightnessDistribution(data);
      const hasGoodBrightness = brightnessDistribution.variance > 500; // Reasonable brightness variation
      
      console.log(`      🔍 Feature analysis:`);
      console.log(`         Color variance: ${colorVariance.toFixed(0)} (${hasColorVariation ? 'good' : 'poor'})`);
      console.log(`         Edge content: ${(edgeContent * 100).toFixed(1)}% (${hasEdgeContent ? 'good' : 'poor'})`);
      console.log(`         Brightness variance: ${brightnessDistribution.variance.toFixed(0)} (${hasGoodBrightness ? 'good' : 'poor'})`);
      
      // Card-like if it has at least 2 of 3 features
      const featureCount = [hasColorVariation, hasEdgeContent, hasGoodBrightness].filter(Boolean).length;
      return featureCount >= 2;
      
    } catch (error) {
      console.warn('Feature analysis failed:', error);
      return true; // Default to true if analysis fails
    }
  }

  /**
   * Calculate color variance in the image
   */
  private calculateColorVariance(data: Uint8ClampedArray): number {
    let rSum = 0, gSum = 0, bSum = 0;
    let rSumSq = 0, gSumSq = 0, bSumSq = 0;
    const pixelCount = data.length / 4;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      rSum += r;
      gSum += g;
      bSum += b;
      
      rSumSq += r * r;
      gSumSq += g * g;
      bSumSq += b * b;
    }
    
    const rMean = rSum / pixelCount;
    const gMean = gSum / pixelCount;
    const bMean = bSum / pixelCount;
    
    const rVar = (rSumSq / pixelCount) - (rMean * rMean);
    const gVar = (gSumSq / pixelCount) - (gMean * gMean);
    const bVar = (bSumSq / pixelCount) - (bMean * bMean);
    
    return (rVar + gVar + bVar) / 3;
  }

  /**
   * Calculate edge content using simple gradient detection
   */
  private calculateEdgeContent(data: Uint8ClampedArray, width: number, height: number): number {
    let edgePixels = 0;
    const threshold = 30; // Edge detection threshold
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        
        // Get current pixel brightness
        const current = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        
        // Get neighboring pixels
        const right = (data[idx + 4] + data[idx + 5] + data[idx + 6]) / 3;
        const bottom = (data[idx + width * 4] + data[idx + width * 4 + 1] + data[idx + width * 4 + 2]) / 3;
        
        // Check for edges
        if (Math.abs(current - right) > threshold || Math.abs(current - bottom) > threshold) {
          edgePixels++;
        }
      }
    }
    
    return edgePixels / ((width - 2) * (height - 2));
  }

  /**
   * Calculate brightness distribution
   */
  private calculateBrightnessDistribution(data: Uint8ClampedArray): { mean: number; variance: number } {
    let sum = 0;
    let sumSq = 0;
    const pixelCount = data.length / 4;
    
    for (let i = 0; i < data.length; i += 4) {
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      sum += brightness;
      sumSq += brightness * brightness;
    }
    
    const mean = sum / pixelCount;
    const variance = (sumSq / pixelCount) - (mean * mean);
    
    return { mean, variance };
  }

  /**
   * Update validation options
   */
  updateOptions(newOptions: Partial<PostProcessValidationOptions>): void {
    this.options = { ...this.options, ...newOptions };
  }

  /**
   * Get current validation options
   */
  getOptions(): PostProcessValidationOptions {
    return { ...this.options };
  }
}
