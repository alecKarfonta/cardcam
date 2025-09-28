import { ExtractedCard } from '../store/slices/cardExtractionSlice';
import { CardDetection } from '../store/slices/inferenceSlice';
import { HighResCardExtractor, HighResExtractionOptions } from './HighResCardExtractor';

export interface DimensionAdjustments {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/**
 * Utility class for re-extracting cards with modified dimensions
 */
export class CardReExtractor {
  private static extractor = new HighResCardExtractor();

  /**
   * Re-extract a card with modified dimensions from the original image
   */
  static async reExtractCard(
    originalImageData: ImageData,
    card: ExtractedCard,
    adjustments: DimensionAdjustments
  ): Promise<ExtractedCard | null> {
    try {
      console.log('🔄 Re-extracting card with dimension adjustments:', adjustments);
      console.log('📊 Original card dimensions:', card.dimensions);
      console.log('🎯 Card ID:', card.id);

      // Get the original detection and extraction metadata
      const originalDetection = card.originalDetection;
      const extractionMeta = card.extractionMetadata;

      if (!extractionMeta) {
        console.error('❌ No extraction metadata found for card');
        return null;
      }

      // Create a modified detection with adjusted bounding box
      const modifiedDetection = this.createModifiedDetection(
        originalDetection,
        extractionMeta,
        adjustments
      );

      if (!modifiedDetection) {
        console.error('❌ Failed to create modified detection');
        return null;
      }

      console.log('📐 Modified detection bbox:', modifiedDetection.boundingBox);

      // Re-extract using the modified detection
      const extractionOptions: Partial<HighResExtractionOptions> = {
        modelInputSize: extractionMeta.modelInputSize,
        paddingRatio: 0.02, // Minimal padding since we're manually adjusting
        minCardSize: 100,
        maxCardSize: 3000,
        enablePerspectiveCorrection: extractionMeta.extractionMethod === 'perspective'
      };

      // Create new extractor instance with updated options
      const reExtractor = new HighResCardExtractor(extractionOptions);
      
      const results = await reExtractor.extractCards(
        originalImageData,
        [modifiedDetection],
        extractionMeta.originalSize
      );

      if (results.length === 0) {
        console.error('❌ Re-extraction failed - no results');
        return null;
      }

      const result = results[0];
      console.log('✅ Re-extraction successful:', result.extractedWidth, 'x', result.extractedHeight);

      // Create updated card with new image data and metadata
      const updatedCard: ExtractedCard = {
        ...card,
        imageData: result.imageData,
        dimensions: {
          width: result.extractedWidth,
          height: result.extractedHeight
        },
        extractionMetadata: {
          ...extractionMeta,
          ...result.metadata,
          // Keep original detection info but note the modification
          cardId: card.id, // Preserve original card ID
        },
        extractedAt: Date.now() // Update extraction timestamp
      };

      return updatedCard;

    } catch (error) {
      console.error('❌ Card re-extraction failed:', error);
      return null;
    }
  }

  /**
   * Create a modified detection with adjusted bounding box
   */
  private static createModifiedDetection(
    originalDetection: CardDetection,
    extractionMeta: any,
    adjustments: DimensionAdjustments
  ): CardDetection | null {
    try {
      const bbox = originalDetection.boundingBox;
      const scalingFactors = extractionMeta.scalingFactors;
      const originalSize = extractionMeta.originalSize;

      console.log('🔍 Original bbox:', bbox);
      console.log('📏 Scaling factors:', scalingFactors);
      console.log('📐 Original image size:', originalSize);

      // Convert adjustments from pixels to normalized coordinates
      // Adjustments are in pixels relative to the extracted card size
      let adjustedBbox;

      if (bbox.x <= 1 && bbox.y <= 1 && bbox.width <= 1 && bbox.height <= 1) {
        // Bbox is in normalized coordinates (0-1)
        console.log('📊 Working with normalized coordinates');
        
        // Convert pixel adjustments to normalized adjustments
        const leftAdjustNorm = adjustments.left / originalSize.width;
        const rightAdjustNorm = adjustments.right / originalSize.width;
        const topAdjustNorm = adjustments.top / originalSize.height;
        const bottomAdjustNorm = adjustments.bottom / originalSize.height;

        adjustedBbox = {
          x: Math.max(0, bbox.x - leftAdjustNorm),
          y: Math.max(0, bbox.y - topAdjustNorm),
          width: Math.min(1 - (bbox.x - leftAdjustNorm), bbox.width + leftAdjustNorm + rightAdjustNorm),
          height: Math.min(1 - (bbox.y - topAdjustNorm), bbox.height + topAdjustNorm + bottomAdjustNorm)
        };
      } else {
        // Bbox is in pixel coordinates relative to model input size
        console.log('📊 Working with pixel coordinates');
        
        // Convert pixel adjustments to model input coordinate adjustments
        const leftAdjustModel = adjustments.left / scalingFactors.x;
        const rightAdjustModel = adjustments.right / scalingFactors.x;
        const topAdjustModel = adjustments.top / scalingFactors.y;
        const bottomAdjustModel = adjustments.bottom / scalingFactors.y;

        adjustedBbox = {
          x: Math.max(0, bbox.x - leftAdjustModel),
          y: Math.max(0, bbox.y - topAdjustModel),
          width: Math.min(extractionMeta.modelInputSize.width - (bbox.x - leftAdjustModel), 
                         bbox.width + leftAdjustModel + rightAdjustModel),
          height: Math.min(extractionMeta.modelInputSize.height - (bbox.y - topAdjustModel), 
                          bbox.height + topAdjustModel + bottomAdjustModel)
        };
      }

      console.log('📐 Adjusted bbox:', adjustedBbox);

      // Validate the adjusted bounding box
      if (adjustedBbox.width <= 0 || adjustedBbox.height <= 0) {
        console.error('❌ Invalid adjusted bounding box dimensions');
        return null;
      }

      // Create modified detection
      const modifiedDetection: CardDetection = {
        ...originalDetection,
        boundingBox: adjustedBbox,
        // If there are corners, we should adjust them too, but for now we'll clear them
        // to force bounding box extraction
        corners: undefined,
        isRotated: false
      };

      return modifiedDetection;

    } catch (error) {
      console.error('❌ Failed to create modified detection:', error);
      return null;
    }
  }

  /**
   * Calculate the expected new dimensions after applying adjustments
   */
  static calculateNewDimensions(
    currentDimensions: { width: number; height: number },
    adjustments: DimensionAdjustments
  ): { width: number; height: number } {
    return {
      width: currentDimensions.width + adjustments.left + adjustments.right,
      height: currentDimensions.height + adjustments.top + adjustments.bottom
    };
  }

  /**
   * Validate that the adjustments are reasonable
   */
  static validateAdjustments(
    currentDimensions: { width: number; height: number },
    adjustments: DimensionAdjustments,
    originalImageSize: { width: number; height: number }
  ): { valid: boolean; reason?: string } {
    const newDimensions = this.calculateNewDimensions(currentDimensions, adjustments);

    // Check minimum size
    if (newDimensions.width < 50 || newDimensions.height < 50) {
      return { valid: false, reason: 'Resulting dimensions would be too small (minimum 50x50)' };
    }

    // Check maximum size (shouldn't exceed original image)
    if (newDimensions.width > originalImageSize.width * 1.2 || 
        newDimensions.height > originalImageSize.height * 1.2) {
      return { valid: false, reason: 'Resulting dimensions would exceed reasonable bounds' };
    }

    // Check individual adjustments aren't too extreme
    const maxAdjustment = Math.max(originalImageSize.width, originalImageSize.height) * 0.3;
    if (Math.abs(adjustments.left) > maxAdjustment || 
        Math.abs(adjustments.right) > maxAdjustment ||
        Math.abs(adjustments.top) > maxAdjustment || 
        Math.abs(adjustments.bottom) > maxAdjustment) {
      return { valid: false, reason: 'Individual adjustments are too large' };
    }

    return { valid: true };
  }
}
