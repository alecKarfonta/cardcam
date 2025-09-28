import { CardDetection, InferenceResult } from '../store/slices/inferenceSlice';

export interface DetectionQuality {
  score: number;
  factors: {
    stability: number;
    confidence: number;
    positioning: number;
    lighting: number;
  };
}

export interface CardPositioning {
  isWellPositioned: boolean;
  feedback: string;
  quality: DetectionQuality;
  recommendedAction?: string;
}

export class CardDetectionSystem {
  private detectionHistory: InferenceResult[] = [];
  private maxHistorySize: number = 10;
  private stabilityThreshold: number = 0.8;
  private confidenceThreshold: number = 0.7;

  constructor(options?: {
    maxHistorySize?: number;
    stabilityThreshold?: number;
    confidenceThreshold?: number;
  }) {
    if (options) {
      this.maxHistorySize = options.maxHistorySize ?? this.maxHistorySize;
      this.stabilityThreshold = options.stabilityThreshold ?? this.stabilityThreshold;
      this.confidenceThreshold = options.confidenceThreshold ?? this.confidenceThreshold;
    }
  }

  addDetectionResult(result: InferenceResult): void {
    this.detectionHistory.unshift(result);
    if (this.detectionHistory.length > this.maxHistorySize) {
      this.detectionHistory = this.detectionHistory.slice(0, this.maxHistorySize);
    }
  }

  analyzeCardPositioning(): CardPositioning {
    if (this.detectionHistory.length === 0) {
      return {
        isWellPositioned: false,
        feedback: 'Searching for cards...',
        quality: this.createEmptyQuality(),
      };
    }

    const latestResult = this.detectionHistory[0];
    
    if (latestResult.detections.length === 0) {
      return {
        isWellPositioned: false,
        feedback: 'No cards detected. Ensure card is visible and well-lit.',
        quality: this.createEmptyQuality(),
        recommendedAction: 'position_card'
      };
    }

    // Analyze the best detection
    const bestDetection = this.getBestDetection(latestResult.detections);
    const quality = this.assessDetectionQuality(bestDetection);

    return this.generatePositioningFeedback(bestDetection, quality);
  }

  private getBestDetection(detections: CardDetection[]): CardDetection {
    return detections.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );
  }

  private assessDetectionQuality(detection: CardDetection): DetectionQuality {
    const stability = this.calculateStability(detection);
    const confidence = detection.confidence;
    const positioning = this.assessPositioning(detection);
    const lighting = this.assessLighting(detection);

    const score = (stability + confidence + positioning + lighting) / 4;

    return {
      score,
      factors: {
        stability,
        confidence,
        positioning,
        lighting,
      },
    };
  }

  private calculateStability(detection: CardDetection): number {
    if (this.detectionHistory.length < 3) return 0.5;

    const recentDetections = this.detectionHistory.slice(0, 5);
    const positions = recentDetections
      .filter(result => result.detections.length > 0)
      .map(result => this.getBestDetection(result.detections).boundingBox);

    if (positions.length < 2) return 0.5;

    // Calculate position variance
    const centerPositions = positions.map(box => ({
      x: box.x + box.width / 2,
      y: box.y + box.height / 2,
    }));

    const avgX = centerPositions.reduce((sum, pos) => sum + pos.x, 0) / centerPositions.length;
    const avgY = centerPositions.reduce((sum, pos) => sum + pos.y, 0) / centerPositions.length;

    const variance = centerPositions.reduce((sum, pos) => {
      return sum + Math.pow(pos.x - avgX, 2) + Math.pow(pos.y - avgY, 2);
    }, 0) / centerPositions.length;

    // Convert variance to stability score (lower variance = higher stability)
    const maxVariance = 10000; // Adjust based on image size
    return Math.max(0, 1 - variance / maxVariance);
  }

  private assessPositioning(detection: CardDetection): number {
    const { boundingBox } = detection;
    const imageWidth = 1920; // TODO: Get from actual image dimensions
    const imageHeight = 1080;

    // Check if card is well-centered and appropriately sized
    const centerX = boundingBox.x + boundingBox.width / 2;
    const centerY = boundingBox.y + boundingBox.height / 2;
    
    const imageCenterX = imageWidth / 2;
    const imageCenterY = imageHeight / 2;

    // Distance from center (normalized)
    const centerDistance = Math.sqrt(
      Math.pow((centerX - imageCenterX) / imageWidth, 2) +
      Math.pow((centerY - imageCenterY) / imageHeight, 2)
    );

    // Size appropriateness (card should occupy reasonable portion of image)
    const cardArea = boundingBox.width * boundingBox.height;
    const imageArea = imageWidth * imageHeight;
    const areaRatio = cardArea / imageArea;
    
    const idealAreaRatio = 0.3; // 30% of image
    const sizeScore = 1 - Math.abs(areaRatio - idealAreaRatio) / idealAreaRatio;

    // Combine center and size scores
    const centerScore = Math.max(0, 1 - centerDistance * 2);
    return (centerScore + Math.max(0, sizeScore)) / 2;
  }

  private assessLighting(detection: CardDetection): number {
    // TODO: Implement actual lighting analysis
    // For now, use confidence as a proxy for lighting quality
    return Math.min(1, detection.confidence * 1.2);
  }

  private generatePositioningFeedback(
    detection: CardDetection,
    quality: DetectionQuality
  ): CardPositioning {
    const { score, factors } = quality;

    if (score > 0.85) {
      return {
        isWellPositioned: true,
        feedback: 'Perfect! Tap to capture',
        quality,
      };
    }

    if (score > 0.7) {
      return {
        isWellPositioned: false,
        feedback: 'Good positioning! Hold steady...',
        quality,
        recommendedAction: 'hold_steady'
      };
    }

    if (score > 0.5) {
      // Determine primary issue
      const lowestFactor = Object.entries(factors).reduce((lowest, [key, value]) => 
        value < lowest.value ? { key, value } : lowest,
        { key: 'confidence', value: 1 }
      );

      const feedbackMap: Record<string, string> = {
        stability: 'Hold the camera steady',
        confidence: 'Move closer to the card',
        positioning: 'Center the card in the frame',
        lighting: 'Improve lighting conditions',
      };

      return {
        isWellPositioned: false,
        feedback: feedbackMap[lowestFactor.key] || 'Adjust card position',
        quality,
        recommendedAction: 'adjust_position'
      };
    }

    return {
      isWellPositioned: false,
      feedback: 'Card detected but needs better positioning',
      quality,
      recommendedAction: 'reposition_card'
    };
  }

  private createEmptyQuality(): DetectionQuality {
    return {
      score: 0,
      factors: {
        stability: 0,
        confidence: 0,
        positioning: 0,
        lighting: 0,
      },
    };
  }

  getDetectionHistory(): InferenceResult[] {
    return [...this.detectionHistory];
  }

  clearHistory(): void {
    this.detectionHistory = [];
  }

  updateThresholds(thresholds: {
    stability?: number;
    confidence?: number;
  }): void {
    if (thresholds.stability !== undefined) {
      this.stabilityThreshold = thresholds.stability;
    }
    if (thresholds.confidence !== undefined) {
      this.confidenceThreshold = thresholds.confidence;
    }
  }
}
