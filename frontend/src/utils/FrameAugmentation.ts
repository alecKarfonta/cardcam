/**
 * Frame Augmentation utility for Test Time Augmentation (TTA)
 * Applies slight perturbations to frames to improve detection robustness
 */

export interface AugmentationConfig {
  /** Enable brightness adjustments */
  enableBrightness?: boolean;
  /** Brightness adjustment range (-0.2 to +0.2) */
  brightnessRange?: [number, number];
  
  /** Enable contrast adjustments */
  enableContrast?: boolean;
  /** Contrast adjustment range (0.8 to 1.2) */
  contrastRange?: [number, number];
  
  /** Enable slight rotation */
  enableRotation?: boolean;
  /** Rotation range in degrees (-2 to +2) */
  rotationRange?: [number, number];
  
  /** Enable slight scaling */
  enableScale?: boolean;
  /** Scale range (0.98 to 1.02) */
  scaleRange?: [number, number];
  
  /** Enable slight translation */
  enableTranslation?: boolean;
  /** Translation range in pixels (-5 to +5) */
  translationRange?: [number, number];
  
  /** Enable noise addition */
  enableNoise?: boolean;
  /** Noise intensity (0 to 10) */
  noiseIntensity?: number;
  
  /** Enable gamma correction */
  enableGamma?: boolean;
  /** Gamma range (0.9 to 1.1) */
  gammaRange?: [number, number];
}

export interface AugmentedFrame {
  imageData: ImageData;
  augmentationType: string;
  parameters: Record<string, number>;
  frameIndex: number; // For tracking in fusion
}

export interface AugmentationResult {
  originalFrame: ImageData;
  augmentedFrames: AugmentedFrame[];
  totalFrames: number;
  augmentationDuration: number;
}

/**
 * Frame Augmentation utility for creating robust detection variations
 */
export class FrameAugmentation {
  private static readonly DEFAULT_CONFIG: Required<AugmentationConfig> = {
    enableBrightness: true,
    brightnessRange: [-0.15, 0.15],
    enableContrast: true,
    contrastRange: [0.85, 1.15],
    enableRotation: true,
    rotationRange: [-1.5, 1.5],
    enableScale: true,
    scaleRange: [0.98, 1.02],
    enableTranslation: true,
    translationRange: [-3, 3],
    enableNoise: true,
    noiseIntensity: 5,
    enableGamma: true,
    gammaRange: [0.92, 1.08]
  };

  /**
   * Generate augmented versions of a frame for robust detection
   */
  static async augmentFrame(
    originalFrame: ImageData,
    config: AugmentationConfig = {},
    options: {
      numAugmentations?: number;
      canvas?: HTMLCanvasElement;
    } = {}
  ): Promise<AugmentationResult> {
    const {
      numAugmentations = 4,
      canvas = document.createElement('canvas')
    } = options;

    const fullConfig = { ...this.DEFAULT_CONFIG, ...config };
    const startTime = performance.now();
    
    console.log(`🎨 Starting frame augmentation: generating ${numAugmentations} variations...`);
    
    const augmentedFrames: AugmentedFrame[] = [];
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas 2D context for frame augmentation');
    }

    // Set canvas size to match original frame
    canvas.width = originalFrame.width;
    canvas.height = originalFrame.height;

    // Generate different types of augmentations
    const augmentationTypes = this.getEnabledAugmentations(fullConfig);
    
    for (let i = 0; i < numAugmentations; i++) {
      // Yield to browser between augmentations to prevent UI blocking
      if (i > 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      const augmentationType = augmentationTypes[i % augmentationTypes.length];
      const augmentedFrame = await this.applyAugmentation(
        originalFrame,
        augmentationType,
        fullConfig,
        canvas,
        ctx,
        i
      );
      
      augmentedFrames.push(augmentedFrame);
    }

    const augmentationDuration = performance.now() - startTime;
    
    console.log(`✅ Frame augmentation complete: ${augmentedFrames.length} variations in ${augmentationDuration.toFixed(2)}ms`);
    console.log(`🎯 Augmentation types used:`, augmentedFrames.map(f => f.augmentationType));

    return {
      originalFrame,
      augmentedFrames,
      totalFrames: augmentedFrames.length + 1, // +1 for original
      augmentationDuration
    };
  }

  /**
   * Get list of enabled augmentation types
   */
  private static getEnabledAugmentations(config: Required<AugmentationConfig>): string[] {
    const types: string[] = [];
    
    if (config.enableBrightness) types.push('brightness');
    if (config.enableContrast) types.push('contrast');
    if (config.enableRotation) types.push('rotation');
    if (config.enableScale) types.push('scale');
    if (config.enableTranslation) types.push('translation');
    if (config.enableNoise) types.push('noise');
    if (config.enableGamma) types.push('gamma');
    
    return types;
  }

  /**
   * Apply a specific augmentation to the frame
   */
  private static async applyAugmentation(
    originalFrame: ImageData,
    augmentationType: string,
    config: Required<AugmentationConfig>,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    frameIndex: number
  ): Promise<AugmentedFrame> {
    // Clear canvas and draw original frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(originalFrame, 0, 0);

    let parameters: Record<string, number> = {};

    switch (augmentationType) {
      case 'brightness':
        parameters = await this.applyBrightnessAugmentation(ctx, canvas, config);
        break;
      case 'contrast':
        parameters = await this.applyContrastAugmentation(ctx, canvas, config);
        break;
      case 'rotation':
        parameters = await this.applyRotationAugmentation(ctx, canvas, config);
        break;
      case 'scale':
        parameters = await this.applyScaleAugmentation(ctx, canvas, config);
        break;
      case 'translation':
        parameters = await this.applyTranslationAugmentation(ctx, canvas, config);
        break;
      case 'noise':
        parameters = await this.applyNoiseAugmentation(ctx, canvas, config);
        break;
      case 'gamma':
        parameters = await this.applyGammaAugmentation(ctx, canvas, config);
        break;
      default:
        console.warn(`Unknown augmentation type: ${augmentationType}`);
    }

    // Get augmented image data
    const augmentedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return {
      imageData: augmentedImageData,
      augmentationType,
      parameters,
      frameIndex: frameIndex + 1000 // Offset to distinguish from original frames
    };
  }

  /**
   * Apply brightness adjustment
   */
  private static async applyBrightnessAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const brightness = this.randomInRange(config.brightnessRange[0], config.brightnessRange[1]);
    
    // Apply brightness using canvas filter
    ctx.filter = `brightness(${1 + brightness})`;
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
    
    return { brightness };
  }

  /**
   * Apply contrast adjustment
   */
  private static async applyContrastAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const contrast = this.randomInRange(config.contrastRange[0], config.contrastRange[1]);
    
    ctx.filter = `contrast(${contrast})`;
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
    
    return { contrast };
  }

  /**
   * Apply slight rotation
   */
  private static async applyRotationAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const rotation = this.randomInRange(config.rotationRange[0], config.rotationRange[1]);
    const rotationRad = (rotation * Math.PI) / 180;
    
    // Save current state
    ctx.save();
    
    // Rotate around center
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotationRad);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
    
    // Draw rotated image
    ctx.drawImage(canvas, 0, 0);
    
    // Restore state
    ctx.restore();
    
    return { rotation };
  }

  /**
   * Apply slight scaling
   */
  private static async applyScaleAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const scale = this.randomInRange(config.scaleRange[0], config.scaleRange[1]);
    
    ctx.save();
    
    // Scale from center
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(scale, scale);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
    
    ctx.drawImage(canvas, 0, 0);
    ctx.restore();
    
    return { scale };
  }

  /**
   * Apply slight translation
   */
  private static async applyTranslationAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const translateX = this.randomInRange(config.translationRange[0], config.translationRange[1]);
    const translateY = this.randomInRange(config.translationRange[0], config.translationRange[1]);
    
    ctx.save();
    ctx.translate(translateX, translateY);
    ctx.drawImage(canvas, 0, 0);
    ctx.restore();
    
    return { translateX, translateY };
  }

  /**
   * Apply noise to the image
   */
  private static async applyNoiseAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const intensity = config.noiseIntensity;
    
    // Add random noise to each pixel
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * intensity;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));     // R
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise)); // G
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise)); // B
      // Alpha channel (i + 3) remains unchanged
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    return { noiseIntensity: intensity };
  }

  /**
   * Apply gamma correction
   */
  private static async applyGammaAugmentation(
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    config: Required<AugmentationConfig>
  ): Promise<Record<string, number>> {
    const gamma = this.randomInRange(config.gammaRange[0], config.gammaRange[1]);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Apply gamma correction
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.pow(data[i] / 255, gamma) * 255;         // R
      data[i + 1] = Math.pow(data[i + 1] / 255, gamma) * 255; // G
      data[i + 2] = Math.pow(data[i + 2] / 255, gamma) * 255; // B
      // Alpha channel remains unchanged
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    return { gamma };
  }

  /**
   * Generate random number in range
   */
  private static randomInRange(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  /**
   * Create a conservative augmentation config for production use
   */
  static createConservativeConfig(): AugmentationConfig {
    return {
      enableBrightness: true,
      brightnessRange: [-0.1, 0.1],
      enableContrast: true,
      contrastRange: [0.9, 1.1],
      enableRotation: true,
      rotationRange: [-1, 1],
      enableScale: false, // Disable scaling to avoid coordinate issues
      enableTranslation: true,
      translationRange: [-2, 2],
      enableNoise: true,
      noiseIntensity: 3,
      enableGamma: true,
      gammaRange: [0.95, 1.05]
    };
  }

  /**
   * Create an aggressive augmentation config for maximum robustness
   */
  static createAggressiveConfig(): AugmentationConfig {
    return {
      enableBrightness: true,
      brightnessRange: [-0.2, 0.2],
      enableContrast: true,
      contrastRange: [0.8, 1.2],
      enableRotation: true,
      rotationRange: [-2, 2],
      enableScale: true,
      scaleRange: [0.97, 1.03],
      enableTranslation: true,
      translationRange: [-5, 5],
      enableNoise: true,
      noiseIntensity: 8,
      enableGamma: true,
      gammaRange: [0.9, 1.1]
    };
  }

  /**
   * Create a minimal augmentation config for fast processing
   */
  static createMinimalConfig(): AugmentationConfig {
    return {
      enableBrightness: true,
      brightnessRange: [-0.05, 0.05],
      enableContrast: true,
      contrastRange: [0.95, 1.05],
      enableRotation: false,
      enableScale: false,
      enableTranslation: false,
      enableNoise: false,
      enableGamma: false
    };
  }
}
