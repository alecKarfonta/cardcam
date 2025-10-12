// Core types for card detection and extraction

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface DetectionQuality {
  isStable: boolean;
  stability: number;
  score: number;
  primaryDetection?: any;
}

export interface CardDetection {
  id: string;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  isRotated: boolean;
  corners?: Array<{ x: number; y: number }>;
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

export interface ExtractedCard {
  id: string;
  imageData: ImageData;
  originalDetection: CardDetection;
  extractedAt: number;
  dimensions: {
    width: number;
    height: number;
  };
  metadata?: {
    cardName?: string;
    setName?: string;
    rarity?: string;
    condition?: string;
  };
  extractionMetadata?: {
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
  qualityScore?: number;
  qualityFactors?: string[];
  validationResult?: PostProcessValidationResult;
}

// Helper function to convert our simple CapturedCard to ExtractedCard format
export function capturedCardToExtractedCard(
  id: string,
  imageDataBase64: string,
  detection: Detection,
  timestamp: number,
  width: number,
  height: number,
  sourceWidth: number,
  sourceHeight: number
): ExtractedCard {
  // Convert base64 to ImageData
  const img = new Image();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  
  return new Promise<ExtractedCard>((resolve) => {
    img.onload = () => {
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, width, height);
      
      resolve({
        id,
        imageData,
        originalDetection: {
          id: `det-${id}`,
          boundingBox: {
            x: detection.x * sourceWidth,
            y: detection.y * sourceHeight,
            width: detection.width * sourceWidth,
            height: detection.height * sourceHeight,
          },
          confidence: detection.confidence,
          isRotated: false
        },
        extractedAt: timestamp,
        dimensions: { width, height },
        extractionMetadata: {
          cardId: id,
          extractionMethod: 'bbox',
          originalSize: { width: sourceWidth, height: sourceHeight },
          modelInputSize: { width: 1088, height: 1088 },
          scalingFactors: { x: 1, y: 1 },
          confidence: detection.confidence,
          paddingApplied: { x: 10, y: 10 },
          isHighResolution: true
        }
      });
    };
    img.src = imageDataBase64;
  }) as any; // TypeScript workaround for Promise in return type
}

