import { CardDetection, InferenceResult } from '../store/slices/inferenceSlice';

export interface AugmentedInferenceResult extends InferenceResult {
  /** Indicates if this result is from an augmented frame */
  isAugmented?: boolean;
  /** Type of augmentation applied (if any) */
  augmentationType?: string;
  /** Augmentation parameters used */
  augmentationParameters?: Record<string, number>;
}

export interface FusedDetection extends CardDetection {
  /** Number of frames this detection appeared in */
  frameCount: number;
  /** Average confidence across all frames */
  averageConfidence: number;
  /** Confidence scores from each frame */
  frameConfidences: number[];
  /** Source frame indices where this detection was found */
  sourceFrames: number[];
  /** Temporal consistency score (0-1, higher = more stable across frames) */
  temporalConsistency: number;
  /** Multi-frame confidence boost applied */
  confidenceBoost: number;
  /** Detection stability metrics */
  stability: {
    positionVariance: number;
    sizeVariance: number;
    confidenceVariance: number;
  };
  /** Number of augmented frames this detection appeared in */
  augmentedFrameCount: number;
  /** Augmentation robustness score (0-1, higher = more robust across augmentations) */
  augmentationRobustness: number;
  /** Types of augmentations where this detection was found */
  augmentationTypes: string[];
}

export interface DetectionFusionResult {
  /** Fused detections with improved confidence and positioning */
  fusedDetections: FusedDetection[];
  /** Original results from each frame for reference */
  originalResults: (InferenceResult | AugmentedInferenceResult)[];
  /** Statistics about the fusion process */
  fusionStats: {
    totalDetections: number;
    fusedDetections: number;
    averageDetectionsPerFrame: number;
    confidenceImprovement: number;
    multiFrameDetections: number;
    singleFrameDetections: number;
    averageTemporalConsistency: number;
    robustnessScore: number;
    augmentedFrameCount: number;
    augmentationRobustnessScore: number;
    averageAugmentationRobustness: number;
  };
}

/**
 * Utility class for fusing detection results from multiple frames
 * to improve robustness and reduce false positives/negatives
 */
export class DetectionFusion {
  private static readonly DEFAULT_IOU_THRESHOLD = 0.5;
  private static readonly DEFAULT_CONFIDENCE_WEIGHT = 0.7;
  private static readonly DEFAULT_POSITION_WEIGHT = 0.3;
  private static readonly MULTI_FRAME_CONFIDENCE_BOOST = 0.15; // Increased from 0.03
  private static readonly TEMPORAL_CONSISTENCY_WEIGHT = 0.2;
  private static readonly MIN_TEMPORAL_CONSISTENCY = 0.6;

  /**
   * Fuse detection results from multiple frames with enhanced robustness
   * Now supports augmented frames for Test Time Augmentation (TTA)
   */
  static fuseDetections(
    results: (InferenceResult | AugmentedInferenceResult)[],
    options: {
      iouThreshold?: number;
      minFrameCount?: number;
      confidenceWeight?: number;
      positionWeight?: number;
      enableTemporalConsistency?: boolean;
      preserveSingleFrameDetections?: boolean;
      multiFrameBoostFactor?: number;
      augmentationBoostFactor?: number;
    } = {}
  ): DetectionFusionResult {
    const {
      iouThreshold = this.DEFAULT_IOU_THRESHOLD,
      minFrameCount = 1,
      confidenceWeight = this.DEFAULT_CONFIDENCE_WEIGHT,
      positionWeight = this.DEFAULT_POSITION_WEIGHT,
      enableTemporalConsistency = true,
      preserveSingleFrameDetections = true,
      multiFrameBoostFactor = this.MULTI_FRAME_CONFIDENCE_BOOST,
      augmentationBoostFactor = 0.1 // Additional boost for augmentation robustness
    } = options;

    console.log(`🔄 Fusing detections from ${results.length} frames (including augmented)...`);
    
    // Separate original and augmented results
    const originalResults = results.filter(r => !(r as AugmentedInferenceResult).isAugmented);
    const augmentedResults = results.filter(r => (r as AugmentedInferenceResult).isAugmented) as AugmentedInferenceResult[];
    
    console.log(`📊 Frame breakdown: ${originalResults.length} original, ${augmentedResults.length} augmented`);
    
    // Collect all detections with frame and augmentation information
    const allDetections: Array<CardDetection & { 
      frameIndex: number; 
      isAugmented: boolean;
      augmentationType?: string;
    }> = [];
    
    results.forEach((result, frameIndex) => {
      const isAugmented = !!(result as AugmentedInferenceResult).isAugmented;
      const augmentationType = (result as AugmentedInferenceResult).augmentationType;
      
      result.detections.forEach(detection => {
        allDetections.push({ 
          ...detection, 
          frameIndex,
          isAugmented,
          augmentationType
        });
      });
    });

    console.log(`📊 Total detections across all frames: ${allDetections.length}`);

    // Group detections that likely represent the same card
    const detectionGroups = this.groupSimilarDetections(allDetections, iouThreshold);
    console.log(`🎯 Grouped into ${detectionGroups.length} unique detections`);

    // Separate multi-frame and single-frame detections
    const multiFrameGroups = detectionGroups.filter(group => group.length > 1);
    const singleFrameGroups = detectionGroups.filter(group => group.length === 1);
    
    console.log(`📊 Multi-frame groups: ${multiFrameGroups.length}, Single-frame groups: ${singleFrameGroups.length}`);

    // Create fused detections
    const fusedDetections: FusedDetection[] = [];
    
    // Process multi-frame detections with enhanced fusion
    for (const group of multiFrameGroups) {
      if (group.length < minFrameCount) continue;

      const fusedDetection = this.createEnhancedFusedDetection(
        group,
        confidenceWeight,
        positionWeight,
        multiFrameBoostFactor,
        enableTemporalConsistency,
        augmentationBoostFactor
      );
      
      fusedDetections.push(fusedDetection);
    }
    
    // Process single-frame detections with different criteria
    if (preserveSingleFrameDetections) {
      for (const group of singleFrameGroups) {
        const detection = group[0];
        
        // Only keep high-confidence single-frame detections
        if (detection.confidence >= 0.7) {
          const fusedDetection = this.createSingleFrameDetection(
            detection,
            results.length
          );
          
          fusedDetections.push(fusedDetection);
        }
      }
    }

    // Calculate enhanced fusion statistics
    const totalDetections = allDetections.length;
    const averageDetectionsPerFrame = totalDetections / results.length;
    const originalAvgConfidence = totalDetections > 0 ? allDetections.reduce((sum, d) => sum + d.confidence, 0) / totalDetections : 0;
    const fusedAvgConfidence = fusedDetections.length > 0 ? fusedDetections.reduce((sum, d) => sum + d.averageConfidence, 0) / fusedDetections.length : 0;
    const confidenceImprovement = fusedDetections.length > 0 ? fusedAvgConfidence - originalAvgConfidence : 0;
    
    const multiFrameDetections = fusedDetections.filter(d => d.frameCount > 1).length;
    const singleFrameDetections = fusedDetections.filter(d => d.frameCount === 1).length;
    const averageTemporalConsistency = fusedDetections.length > 0 ? 
      fusedDetections.reduce((sum, d) => sum + d.temporalConsistency, 0) / fusedDetections.length : 0;
    
    // Calculate augmentation statistics
    const augmentedFrameCount = augmentedResults.length;
    const averageAugmentationRobustness = fusedDetections.length > 0 ?
      fusedDetections.reduce((sum, d) => sum + d.augmentationRobustness, 0) / fusedDetections.length : 0;
    const augmentationRobustnessScore = this.calculateAugmentationRobustnessScore(fusedDetections);
    
    // Calculate overall robustness score
    const robustnessScore = this.calculateRobustnessScore(fusedDetections, results.length);

    const fusionStats = {
      totalDetections,
      fusedDetections: fusedDetections.length,
      averageDetectionsPerFrame,
      confidenceImprovement,
      multiFrameDetections,
      singleFrameDetections,
      averageTemporalConsistency,
      robustnessScore,
      augmentedFrameCount,
      augmentationRobustnessScore,
      averageAugmentationRobustness
    };

    console.log(`✅ Enhanced fusion complete:`, fusionStats);
    console.log(`🎯 Multi-frame detections: ${multiFrameDetections} (${((multiFrameDetections / Math.max(fusedDetections.length, 1)) * 100).toFixed(1)}%)`);
    console.log(`📊 Average temporal consistency: ${(averageTemporalConsistency * 100).toFixed(1)}%`);
    console.log(`🛡️ Robustness score: ${(robustnessScore * 100).toFixed(1)}%`);
    if (augmentedFrameCount > 0) {
      console.log(`🎨 Augmentation frames: ${augmentedFrameCount}, robustness: ${(augmentationRobustnessScore * 100).toFixed(1)}%`);
      console.log(`🔄 Average augmentation robustness: ${(averageAugmentationRobustness * 100).toFixed(1)}%`);
    }

    return {
      fusedDetections,
      originalResults: results,
      fusionStats
    };
  }

  /**
   * Group detections that likely represent the same card across frames
   */
  private static groupSimilarDetections(
    detections: Array<CardDetection & { 
      frameIndex: number; 
      isAugmented: boolean;
      augmentationType?: string;
    }>,
    iouThreshold: number
  ): Array<Array<CardDetection & { 
    frameIndex: number; 
    isAugmented: boolean;
    augmentationType?: string;
  }>> {
    const groups: Array<Array<CardDetection & { 
      frameIndex: number; 
      isAugmented: boolean;
      augmentationType?: string;
    }>> = [];
    const processed = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (processed.has(i)) continue;

      const group = [detections[i]];
      processed.add(i);

      // Find all similar detections
      for (let j = i + 1; j < detections.length; j++) {
        if (processed.has(j)) continue;

        const iou = this.calculateIOU(detections[i], detections[j]);
        if (iou >= iouThreshold) {
          group.push(detections[j]);
          processed.add(j);
        }
      }

      groups.push(group);
    }

    return groups;
  }

  /**
   * Calculate Intersection over Union (IoU) between two detections
   */
  private static calculateIOU(detection1: CardDetection, detection2: CardDetection): number {
    const box1 = detection1.boundingBox;
    const box2 = detection2.boundingBox;

    // Calculate intersection area
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    if (x2 <= x1 || y2 <= y1) {
      return 0; // No intersection
    }

    const intersectionArea = (x2 - x1) * (y2 - y1);

    // Calculate union area
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const unionArea = area1 + area2 - intersectionArea;

    return unionArea > 0 ? intersectionArea / unionArea : 0;
  }

  /**
   * Create a fused detection from a group of similar detections (legacy method)
   */
  private static createFusedDetection(
    group: Array<CardDetection & { frameIndex: number }>,
    confidenceWeight: number,
    positionWeight: number
  ): FusedDetection {
    // Convert to enhanced format with default augmentation values
    const enhancedGroup = group.map(detection => ({
      ...detection,
      isAugmented: false,
      augmentationType: undefined
    }));
    
    // Use enhanced method with default parameters
    return this.createEnhancedFusedDetection(
      enhancedGroup,
      confidenceWeight,
      positionWeight,
      this.MULTI_FRAME_CONFIDENCE_BOOST,
      true
    );
  }

  /**
   * Create an enhanced fused detection from a group of similar detections
   */
  private static createEnhancedFusedDetection(
    group: Array<CardDetection & { 
      frameIndex: number; 
      isAugmented: boolean;
      augmentationType?: string;
    }>,
    confidenceWeight: number,
    positionWeight: number,
    multiFrameBoostFactor: number,
    enableTemporalConsistency: boolean,
    augmentationBoostFactor: number = 0.1
  ): FusedDetection {
    // Calculate weighted average position
    const totalWeight = group.reduce((sum, d) => sum + d.confidence, 0);
    
    let avgX = 0, avgY = 0, avgWidth = 0, avgHeight = 0;
    let avgCorners: Array<{ x: number; y: number }> | undefined;

    // Initialize corners array if any detection has corners
    const hasRotatedDetections = group.some(d => d.corners);
    if (hasRotatedDetections) {
      avgCorners = [
        { x: 0, y: 0 },
        { x: 0, y: 0 },
        { x: 0, y: 0 },
        { x: 0, y: 0 }
      ];
    }

    for (const detection of group) {
      const weight = detection.confidence / totalWeight;
      
      avgX += detection.boundingBox.x * weight;
      avgY += detection.boundingBox.y * weight;
      avgWidth += detection.boundingBox.width * weight;
      avgHeight += detection.boundingBox.height * weight;

      // Average corners if available
      if (avgCorners && detection.corners && detection.corners.length === 4) {
        for (let i = 0; i < 4; i++) {
          avgCorners[i].x += detection.corners[i].x * weight;
          avgCorners[i].y += detection.corners[i].y * weight;
        }
      }
    }

    // Calculate enhanced confidence metrics
    const frameConfidences = group.map(d => d.confidence);
    const averageConfidence = frameConfidences.reduce((sum, c) => sum + c, 0) / frameConfidences.length;
    const maxConfidence = Math.max(...frameConfidences);
    const minConfidence = Math.min(...frameConfidences);
    
    // Calculate temporal consistency and stability
    const temporalConsistency = this.calculateTemporalConsistency(group);
    const stability = this.calculateStabilityMetrics(group);
    
    // Calculate augmentation metrics
    const augmentedFrames = group.filter(d => d.isAugmented);
    const augmentedFrameCount = augmentedFrames.length;
    const augmentationTypes = Array.from(new Set(augmentedFrames.map(d => d.augmentationType).filter(Boolean)));
    const augmentationRobustness = this.calculateAugmentationRobustness(group);
    
    // Enhanced confidence boosting for multi-frame detections
    const frameCountBoost = Math.min(0.25, (group.length - 1) * multiFrameBoostFactor);
    const consistencyBoost = enableTemporalConsistency ? temporalConsistency * 0.1 : 0;
    const augmentationBoost = augmentedFrameCount > 0 ? augmentationRobustness * augmentationBoostFactor : 0;
    const confidenceBoost = frameCountBoost + consistencyBoost + augmentationBoost;
    
    // Use weighted combination of max and average confidence
    const baseConfidence = (maxConfidence * 0.7) + (averageConfidence * 0.3);
    const boostedConfidence = Math.min(1.0, baseConfidence + confidenceBoost);

    // Use the detection with highest confidence as the base
    const baseDetection = group.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );

    const fusedDetection: FusedDetection = {
      ...baseDetection,
      boundingBox: {
        x: Math.round(avgX),
        y: Math.round(avgY),
        width: Math.round(avgWidth),
        height: Math.round(avgHeight)
      },
      confidence: boostedConfidence,
      corners: avgCorners,
      frameCount: group.length,
      averageConfidence,
      frameConfidences,
      sourceFrames: group.map(d => d.frameIndex).sort(),
      temporalConsistency,
      confidenceBoost,
      stability,
      augmentedFrameCount,
      augmentationRobustness,
      augmentationTypes: augmentationTypes as string[],
      id: `fused_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };

    return fusedDetection;
  }

  /**
   * Filter fused detections based on enhanced quality criteria
   */
  static filterHighQualityDetections(
    fusedDetections: FusedDetection[],
    options: {
      minFrameCount?: number;
      minAverageConfidence?: number;
      maxConfidenceVariance?: number;
      minTemporalConsistency?: number;
      prioritizeMultiFrame?: boolean;
    } = {}
  ): FusedDetection[] {
    const {
      minFrameCount = 1, // Allow single-frame detections by default
      minAverageConfidence = 0.25,
      maxConfidenceVariance = 0.4,
      minTemporalConsistency = 0.3,
      prioritizeMultiFrame = true
    } = options;

    const filtered = fusedDetections.filter(detection => {
      // Must appear in minimum number of frames
      if (detection.frameCount < minFrameCount) {
        return false;
      }

      // Must have minimum average confidence
      if (detection.averageConfidence < minAverageConfidence) {
        return false;
      }

      // Confidence variance shouldn't be too high (indicates inconsistent detection)
      if (detection.stability.confidenceVariance > maxConfidenceVariance) {
        return false;
      }

      // Multi-frame detections should meet temporal consistency requirements
      if (detection.frameCount > 1 && detection.temporalConsistency < minTemporalConsistency) {
        return false;
      }

      return true;
    });

    // Sort by quality if prioritizing multi-frame detections
    if (prioritizeMultiFrame) {
      return filtered.sort((a, b) => {
        // First priority: multi-frame detections
        if (a.frameCount !== b.frameCount) {
          return b.frameCount - a.frameCount;
        }
        // Second priority: confidence with boost
        return b.confidence - a.confidence;
      });
    }

    return filtered;
  }

  /**
   * Calculate variance of an array of numbers
   */
  private static calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
  }

  /**
   * Create a single-frame detection with appropriate metadata
   */
  private static createSingleFrameDetection(
    detection: CardDetection & { 
      frameIndex: number;
      isAugmented: boolean;
      augmentationType?: string;
    },
    totalFrames: number
  ): FusedDetection {
    return {
      ...detection,
      frameCount: 1,
      averageConfidence: detection.confidence,
      frameConfidences: [detection.confidence],
      sourceFrames: [detection.frameIndex],
      temporalConsistency: 0.5, // Neutral score for single-frame
      confidenceBoost: 0,
      stability: {
        positionVariance: 0,
        sizeVariance: 0,
        confidenceVariance: 0
      },
      augmentedFrameCount: detection.isAugmented ? 1 : 0,
      augmentationRobustness: detection.isAugmented ? 0.5 : 0,
      augmentationTypes: detection.augmentationType ? [detection.augmentationType] : [] as string[],
      id: `single_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
  }

  /**
   * Calculate augmentation robustness score for a group of detections
   */
  private static calculateAugmentationRobustness(
    group: Array<CardDetection & { 
      frameIndex: number; 
      isAugmented: boolean;
      augmentationType?: string;
    }>
  ): number {
    const augmentedFrames = group.filter(d => d.isAugmented);
    
    if (augmentedFrames.length === 0) return 0;
    
    // Count unique augmentation types
    const uniqueAugmentationTypes = new Set(
      augmentedFrames.map(d => d.augmentationType).filter(Boolean)
    ).size;
    
    // Calculate robustness based on:
    // 1. Ratio of augmented frames to total frames
    // 2. Diversity of augmentation types
    // 3. Consistency of detection across augmentations
    
    const augmentationRatio = augmentedFrames.length / group.length;
    const diversityScore = Math.min(1.0, uniqueAugmentationTypes / 4); // Normalize to max 4 types
    
    // Confidence consistency across augmented frames
    const augmentedConfidences = augmentedFrames.map(d => d.confidence);
    const avgAugmentedConfidence = augmentedConfidences.reduce((sum, c) => sum + c, 0) / augmentedConfidences.length;
    const augmentedVariance = this.calculateVariance(augmentedConfidences);
    const consistencyScore = Math.max(0, 1 - (augmentedVariance / avgAugmentedConfidence));
    
    return (augmentationRatio * 0.4) + (diversityScore * 0.3) + (consistencyScore * 0.3);
  }

  /**
   * Calculate temporal consistency score for a group of detections
   */
  private static calculateTemporalConsistency(
    group: Array<CardDetection & { frameIndex: number }>
  ): number {
    if (group.length <= 1) return 0.5;

    // Check if detections appear in consecutive or well-distributed frames
    const frameIndices = group.map(d => d.frameIndex).sort();
    const frameSpread = frameIndices[frameIndices.length - 1] - frameIndices[0];
    const expectedSpread = group.length - 1;
    
    // Temporal distribution score (better if spread across frames)
    const distributionScore = Math.min(1.0, frameSpread / Math.max(expectedSpread, 1));
    
    // Confidence consistency score
    const confidences = group.map(d => d.confidence);
    const avgConfidence = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
    const confidenceVariance = this.calculateVariance(confidences);
    const consistencyScore = Math.max(0, 1 - (confidenceVariance / avgConfidence));
    
    return (distributionScore * 0.4) + (consistencyScore * 0.6);
  }

  /**
   * Calculate stability metrics for a group of detections
   */
  private static calculateStabilityMetrics(
    group: Array<CardDetection & { frameIndex: number }>
  ): { positionVariance: number; sizeVariance: number; confidenceVariance: number } {
    if (group.length <= 1) {
      return { positionVariance: 0, sizeVariance: 0, confidenceVariance: 0 };
    }

    // Position variance (center points)
    const centerX = group.map(d => d.boundingBox.x + d.boundingBox.width / 2);
    const centerY = group.map(d => d.boundingBox.y + d.boundingBox.height / 2);
    const positionVariance = (this.calculateVariance(centerX) + this.calculateVariance(centerY)) / 2;

    // Size variance
    const areas = group.map(d => d.boundingBox.width * d.boundingBox.height);
    const sizeVariance = this.calculateVariance(areas);

    // Confidence variance
    const confidences = group.map(d => d.confidence);
    const confidenceVariance = this.calculateVariance(confidences);

    return { positionVariance, sizeVariance, confidenceVariance };
  }

  /**
   * Calculate overall robustness score for the fusion result
   */
  private static calculateRobustnessScore(
    fusedDetections: FusedDetection[],
    totalFrames: number
  ): number {
    if (fusedDetections.length === 0) return 0;

    // Multi-frame detection ratio (higher is better)
    const multiFrameRatio = fusedDetections.filter(d => d.frameCount > 1).length / fusedDetections.length;
    
    // Average temporal consistency
    const avgConsistency = fusedDetections.reduce((sum, d) => sum + d.temporalConsistency, 0) / fusedDetections.length;
    
    // Average confidence boost from fusion
    const avgBoost = fusedDetections.reduce((sum, d) => sum + d.confidenceBoost, 0) / fusedDetections.length;
    
    // Combine metrics for overall robustness
    return (multiFrameRatio * 0.4) + (avgConsistency * 0.4) + (Math.min(avgBoost / 0.2, 1) * 0.2);
  }

  /**
   * Calculate augmentation robustness score for the fusion result
   */
  private static calculateAugmentationRobustnessScore(
    fusedDetections: FusedDetection[]
  ): number {
    if (fusedDetections.length === 0) return 0;

    // Detections with augmented frames ratio
    const augmentedDetectionRatio = fusedDetections.filter(d => d.augmentedFrameCount > 0).length / fusedDetections.length;
    
    // Average augmentation robustness
    const avgAugmentationRobustness = fusedDetections.reduce((sum, d) => sum + d.augmentationRobustness, 0) / fusedDetections.length;
    
    // Augmentation diversity (unique augmentation types across all detections)
    const allAugmentationTypes = new Set(
      fusedDetections.flatMap(d => d.augmentationTypes)
    ).size;
    const diversityScore = Math.min(1.0, allAugmentationTypes / 6); // Normalize to max 6 types
    
    return (augmentedDetectionRatio * 0.5) + (avgAugmentationRobustness * 0.3) + (diversityScore * 0.2);
  }

  /**
   * Get detailed fusion analysis for debugging
   */
  static analyzeFusionQuality(fusionResult: DetectionFusionResult): {
    summary: string;
    recommendations: string[];
    qualityScore: number;
  } {
    const { fusedDetections, fusionStats } = fusionResult;
    const multiFrameCount = fusionStats.multiFrameDetections;
    const singleFrameCount = fusionStats.singleFrameDetections;
    const robustness = fusionStats.robustnessScore;
    
    let qualityScore = 0;
    const recommendations: string[] = [];
    
    // Analyze multi-frame detection ratio
    const multiFrameRatio = multiFrameCount / Math.max(fusedDetections.length, 1);
    if (multiFrameRatio > 0.7) {
      qualityScore += 40;
    } else if (multiFrameRatio > 0.4) {
      qualityScore += 25;
      recommendations.push('Consider improving camera stability for more consistent detections');
    } else {
      qualityScore += 10;
      recommendations.push('Low multi-frame detection ratio - check for camera movement or lighting issues');
    }
    
    // Analyze temporal consistency
    if (fusionStats.averageTemporalConsistency > 0.8) {
      qualityScore += 30;
    } else if (fusionStats.averageTemporalConsistency > 0.6) {
      qualityScore += 20;
      recommendations.push('Moderate temporal consistency - ensure steady camera positioning');
    } else {
      qualityScore += 5;
      recommendations.push('Low temporal consistency - significant variation between frames detected');
    }
    
    // Analyze confidence improvement
    if (fusionStats.confidenceImprovement > 0.1) {
      qualityScore += 20;
    } else if (fusionStats.confidenceImprovement > 0.05) {
      qualityScore += 10;
    } else {
      recommendations.push('Limited confidence improvement from fusion - check detection quality');
    }
    
    // Analyze robustness score
    if (fusionStats.robustnessScore > 0.8) {
      qualityScore += 10;
    } else if (fusionStats.robustnessScore < 0.4) {
      recommendations.push('Low robustness score - consider adjusting fusion parameters');
    }
    
    const summary = `Fusion Quality: ${qualityScore}/100 - ${multiFrameCount} multi-frame, ${singleFrameCount} single-frame detections. Robustness: ${(fusionStats.robustnessScore * 100).toFixed(1)}%`;
    
    return { summary, recommendations, qualityScore };
  }

  /**
   * Convert fused detections back to regular CardDetection format
   */
  static convertToCardDetections(fusedDetections: FusedDetection[]): CardDetection[] {
    return fusedDetections.map(fused => ({
      id: fused.id,
      boundingBox: fused.boundingBox,
      confidence: fused.confidence,
      isRotated: fused.isRotated,
      corners: fused.corners
    }));
  }
}
