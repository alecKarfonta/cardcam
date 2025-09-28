import { CardDetection } from '../store/slices/inferenceSlice';

export interface CropResult {
  imageData: ImageData;
  croppedWidth: number;
  croppedHeight: number;
}

export class CardCropper {
  private static readonly PADDING_RATIO = 0.1; // 10% padding around detected card for better extraction
  private static readonly MIN_CARD_SIZE = 50; // Minimum card size in pixels
  private static readonly MAX_CARD_SIZE = 2000; // Maximum card size in pixels

  /**
   * Extract individual cards from an image using detection results
   */
  static extractCards(
    sourceImageData: ImageData,
    detections: CardDetection[]
  ): CropResult[] {
    const results: CropResult[] = [];

    for (const detection of detections) {
      try {
        const cropResult = this.extractSingleCard(sourceImageData, detection);
        if (cropResult) {
          results.push(cropResult);
        }
      } catch (error) {
        console.warn('Failed to extract card:', detection.id, error);
      }
    }

    return results;
  }

  /**
   * Extract a single card from the source image
   */
  private static extractSingleCard(
    sourceImageData: ImageData,
    detection: CardDetection
  ): CropResult | null {
    if (detection.isRotated && detection.corners && detection.corners.length === 4) {
      return this.extractRotatedCard(sourceImageData, detection);
    } else {
      return this.extractAxisAlignedCard(sourceImageData, detection);
    }
  }

  /**
   * Extract a rotated card using oriented bounding box corners
   */
  private static extractRotatedCard(
    sourceImageData: ImageData,
    detection: CardDetection
  ): CropResult | null {
    if (!detection.corners || detection.corners.length !== 4) {
      return null;
    }

    const corners = detection.corners.map(corner => ({
      x: Math.round(corner.x),
      y: Math.round(corner.y)
    }));

    // Calculate the bounding rectangle of the rotated card
    const minX = Math.min(...corners.map(c => c.x));
    const maxX = Math.max(...corners.map(c => c.x));
    const minY = Math.min(...corners.map(c => c.y));
    const maxY = Math.max(...corners.map(c => c.y));

    const originalWidth = maxX - minX + 1;
    const originalHeight = maxY - minY + 1;

    // Add padding while respecting image boundaries
    const paddingX = Math.round(originalWidth * this.PADDING_RATIO);
    const paddingY = Math.round(originalHeight * this.PADDING_RATIO);

    // Calculate padded coordinates, ensuring we stay within image bounds
    const paddedMinX = Math.max(0, minX - paddingX);
    const paddedMaxX = Math.min(sourceImageData.width - 1, maxX + paddingX);
    const paddedMinY = Math.max(0, minY - paddingY);
    const paddedMaxY = Math.min(sourceImageData.height - 1, maxY + paddingY);

    const paddedWidth = paddedMaxX - paddedMinX + 1;
    const paddedHeight = paddedMaxY - paddedMinY + 1;

    if (paddedWidth < this.MIN_CARD_SIZE || paddedHeight < this.MIN_CARD_SIZE) {
      console.warn('Card too small to extract:', paddedWidth, 'x', paddedHeight);
      return null;
    }

    // For now, extract the bounding rectangle with padding
    // TODO: Implement proper perspective correction for rotated cards
    return this.cropRectangle(sourceImageData, paddedMinX, paddedMinY, paddedWidth, paddedHeight);
  }

  /**
   * Extract an axis-aligned card using regular bounding box
   */
  private static extractAxisAlignedCard(
    sourceImageData: ImageData,
    detection: CardDetection
  ): CropResult | null {
    const bbox = detection.boundingBox;
    
    // Convert normalized coordinates to pixel coordinates if needed
    let x = bbox.x;
    let y = bbox.y;
    let width = bbox.width;
    let height = bbox.height;

    // Check if coordinates are normalized (0-1 range)
    if (x <= 1 && y <= 1 && width <= 1 && height <= 1) {
      x = Math.round(x * sourceImageData.width);
      y = Math.round(y * sourceImageData.height);
      width = Math.round(width * sourceImageData.width);
      height = Math.round(height * sourceImageData.height);
    } else {
      x = Math.round(x);
      y = Math.round(y);
      width = Math.round(width);
      height = Math.round(height);
    }

    // Calculate padding amounts
    const paddingX = Math.round(width * this.PADDING_RATIO);
    const paddingY = Math.round(height * this.PADDING_RATIO);

    // Calculate desired padded coordinates
    const desiredX = x - paddingX;
    const desiredY = y - paddingY;
    const desiredWidth = width + 2 * paddingX;
    const desiredHeight = height + 2 * paddingY;

    // Clamp to image boundaries while preserving as much padding as possible
    const finalX = Math.max(0, desiredX);
    const finalY = Math.max(0, desiredY);
    const maxWidth = sourceImageData.width - finalX;
    const maxHeight = sourceImageData.height - finalY;
    const finalWidth = Math.min(desiredWidth, maxWidth);
    const finalHeight = Math.min(desiredHeight, maxHeight);

    // If we're at the left/top edge, try to extend right/down to maintain padding
    const actualWidth = Math.min(finalWidth, sourceImageData.width - finalX);
    const actualHeight = Math.min(finalHeight, sourceImageData.height - finalY);

    // Validate dimensions
    if (actualWidth < this.MIN_CARD_SIZE || actualHeight < this.MIN_CARD_SIZE) {
      console.warn('Card too small to extract:', actualWidth, 'x', actualHeight);
      return null;
    }

    if (actualWidth > this.MAX_CARD_SIZE || actualHeight > this.MAX_CARD_SIZE) {
      console.warn('Card too large, might be false detection:', actualWidth, 'x', actualHeight);
      return null;
    }

    return this.cropRectangle(sourceImageData, finalX, finalY, actualWidth, actualHeight);
  }

  /**
   * Calculate padded coordinates that respect image boundaries
   */
  private static calculatePaddedBounds(
    x: number,
    y: number,
    width: number,
    height: number,
    paddingRatio: number,
    imageWidth: number,
    imageHeight: number
  ): { x: number; y: number; width: number; height: number } {
    const paddingX = Math.round(width * paddingRatio);
    const paddingY = Math.round(height * paddingRatio);

    // Calculate desired padded coordinates
    const desiredX = x - paddingX;
    const desiredY = y - paddingY;
    const desiredWidth = width + 2 * paddingX;
    const desiredHeight = height + 2 * paddingY;

    // Clamp to image boundaries
    const finalX = Math.max(0, desiredX);
    const finalY = Math.max(0, desiredY);
    const finalWidth = Math.min(desiredWidth, imageWidth - finalX);
    const finalHeight = Math.min(desiredHeight, imageHeight - finalY);

    return {
      x: finalX,
      y: finalY,
      width: finalWidth,
      height: finalHeight
    };
  }

  /**
   * Crop a rectangle from the source image
   */
  private static cropRectangle(
    sourceImageData: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
  ): CropResult | null {
    // Ensure coordinates are within bounds
    x = Math.max(0, Math.min(x, sourceImageData.width - 1));
    y = Math.max(0, Math.min(y, sourceImageData.height - 1));
    width = Math.min(width, sourceImageData.width - x);
    height = Math.min(height, sourceImageData.height - y);

    if (width <= 0 || height <= 0) {
      return null;
    }

    // Create a canvas to extract the cropped region
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.error('Failed to get canvas context for cropping');
      return null;
    }

    canvas.width = width;
    canvas.height = height;

    // Create ImageData for the source
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) {
      console.error('Failed to get temporary canvas context');
      return null;
    }

    tempCanvas.width = sourceImageData.width;
    tempCanvas.height = sourceImageData.height;
    tempCtx.putImageData(sourceImageData, 0, 0);

    // Draw the cropped region
    ctx.drawImage(
      tempCanvas,
      x, y, width, height,  // Source rectangle
      0, 0, width, height   // Destination rectangle
    );

    // Get the cropped image data
    const croppedImageData = ctx.getImageData(0, 0, width, height);

    return {
      imageData: croppedImageData,
      croppedWidth: width,
      croppedHeight: height
    };
  }

  /**
   * Apply perspective correction to a rotated card (future enhancement)
   */
  private static applyPerspectiveCorrection(
    sourceImageData: ImageData,
    corners: Array<{ x: number; y: number }>
  ): CropResult | null {
    // TODO: Implement perspective correction using transformation matrix
    // This would straighten rotated cards to a standard rectangular format
    console.warn('Perspective correction not yet implemented');
    return null;
  }

  /**
   * Enhance card image quality (future enhancement)
   */
  static enhanceCardImage(imageData: ImageData): ImageData {
    // TODO: Implement image enhancement
    // - Contrast adjustment
    // - Sharpening
    // - Noise reduction
    // - Color correction
    return imageData;
  }

  /**
   * Validate if a detection represents a valid card
   */
  static isValidCardDetection(detection: CardDetection): boolean {
    // Check confidence threshold
    if (detection.confidence < 0.3) {
      return false;
    }

    // Check bounding box dimensions
    const bbox = detection.boundingBox;
    const width = bbox.width <= 1 ? bbox.width * 1920 : bbox.width; // Assume max 1920px width
    const height = bbox.height <= 1 ? bbox.height * 1080 : bbox.height; // Assume max 1080px height

    if (width < this.MIN_CARD_SIZE || height < this.MIN_CARD_SIZE) {
      return false;
    }

    // Check aspect ratio (trading cards are typically rectangular)
    const aspectRatio = width / height;
    if (aspectRatio < 0.4 || aspectRatio > 2.5) {
      return false;
    }

    return true;
  }
}
