/**
 * Utility for capturing multiple frames from a video element
 * with precise timing control for robust card detection
 */
export interface CapturedFrame {
  imageData: ImageData;
  timestamp: number;
  frameIndex: number;
}

export interface MultiFrameCaptureOptions {
  /** Number of frames to capture (default: 3) */
  frameCount?: number;
  /** Delay between frame captures in milliseconds (default: 100ms) */
  frameDelay?: number;
  /** Canvas to use for frame capture (will create one if not provided) */
  canvas?: HTMLCanvasElement;
}

export interface MultiFrameCaptureResult {
  frames: CapturedFrame[];
  /** Index of the middle frame (used for rendering) */
  middleFrameIndex: number;
  /** Total capture duration in milliseconds */
  captureDuration: number;
}

/**
 * Multi-frame capture utility for improved card detection robustness
 */
export class MultiFrameCapture {
  /**
   * Capture multiple frames from a video element with timing control
   */
  static async captureFrames(
    video: HTMLVideoElement,
    options: MultiFrameCaptureOptions = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      frameCount = 3,
      frameDelay = 100,
      canvas = document.createElement('canvas')
    } = options;

    console.log(`📸 Starting multi-frame capture: ${frameCount} frames with ${frameDelay}ms delay`);
    
    const startTime = performance.now();
    const frames: CapturedFrame[] = [];
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas 2D context for frame capture');
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    for (let i = 0; i < frameCount; i++) {
      const frameStartTime = performance.now();
      
      // Capture current frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      const frame: CapturedFrame = {
        imageData,
        timestamp: frameStartTime,
        frameIndex: i
      };
      
      frames.push(frame);
      
      console.log(`📷 Captured frame ${i + 1}/${frameCount} (${canvas.width}x${canvas.height})`);
      
      // Wait before capturing next frame (except for the last frame)
      if (i < frameCount - 1) {
        await this.delay(frameDelay);
      }
    }

    const captureDuration = performance.now() - startTime;
    const middleFrameIndex = Math.floor(frameCount / 2);

    console.log(`✅ Multi-frame capture complete: ${frameCount} frames in ${captureDuration.toFixed(2)}ms`);
    console.log(`🎯 Middle frame index: ${middleFrameIndex}`);

    return {
      frames,
      middleFrameIndex,
      captureDuration
    };
  }

  /**
   * Capture multiple frames as quickly as possible with minimal delay
   */
  static async captureFramesRapid(
    video: HTMLVideoElement,
    options: Omit<MultiFrameCaptureOptions, 'frameDelay'> & {
      /** Minimum delay between frames in milliseconds (default: 33ms ≈ 30fps) */
      minDelay?: number;
    } = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      frameCount = 3,
      minDelay = 33, // ~30fps minimum to allow for video frame updates
      canvas = document.createElement('canvas')
    } = options;

    console.log(`⚡ Starting rapid multi-frame capture: ${frameCount} frames with ${minDelay}ms minimum delay`);
    
    const startTime = performance.now();
    const frames: CapturedFrame[] = [];
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas 2D context for frame capture');
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Capture all frames as quickly as possible
    for (let i = 0; i < frameCount; i++) {
      const frameStartTime = performance.now();
      
      // Capture current frame immediately
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      const frame: CapturedFrame = {
        imageData,
        timestamp: frameStartTime,
        frameIndex: i
      };
      
      frames.push(frame);
      
      console.log(`⚡ Rapid captured frame ${i + 1}/${frameCount} at ${(frameStartTime - startTime).toFixed(1)}ms`);
      
      // Only wait the minimum delay to allow video to update (except for last frame)
      if (i < frameCount - 1) {
        await this.delay(minDelay);
      }
    }

    const captureDuration = performance.now() - startTime;
    const middleFrameIndex = Math.floor(frameCount / 2);

    console.log(`✅ Rapid multi-frame capture complete: ${frameCount} frames in ${captureDuration.toFixed(2)}ms`);
    console.log(`🎯 Middle frame index: ${middleFrameIndex}`);
    console.log(`📊 Average time between frames: ${(captureDuration / (frameCount - 1)).toFixed(1)}ms`);

    return {
      frames,
      middleFrameIndex,
      captureDuration
    };
  }

  /**
   * Capture multiple frames as fast as possible with NO delays whatsoever
   */
  static async captureFramesInstant(
    video: HTMLVideoElement,
    options: Omit<MultiFrameCaptureOptions, 'frameDelay'> = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      frameCount = 3,
      canvas = document.createElement('canvas')
    } = options;

    console.log(`🚀 Starting INSTANT multi-frame capture: ${frameCount} frames with NO delays`);
    
    const startTime = performance.now();
    const frames: CapturedFrame[] = [];
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Failed to get canvas 2D context for frame capture');
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Capture all frames instantly - no delays whatsoever
    for (let i = 0; i < frameCount; i++) {
      const frameStartTime = performance.now();
      
      // Capture current frame immediately
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      const frame: CapturedFrame = {
        imageData,
        timestamp: frameStartTime,
        frameIndex: i
      };
      
      frames.push(frame);
      
      console.log(`🚀 INSTANT captured frame ${i + 1}/${frameCount} at ${(frameStartTime - startTime).toFixed(2)}ms`);
      
      // NO DELAYS AT ALL - capture as fast as possible
    }

    const captureDuration = performance.now() - startTime;
    const middleFrameIndex = Math.floor(frameCount / 2);

    console.log(`✅ INSTANT multi-frame capture complete: ${frameCount} frames in ${captureDuration.toFixed(2)}ms`);
    console.log(`🎯 Middle frame index: ${middleFrameIndex}`);
    if (frameCount > 1) {
      console.log(`📊 Average time between frames: ${(captureDuration / (frameCount - 1)).toFixed(2)}ms`);
    }

    return {
      frames,
      middleFrameIndex,
      captureDuration
    };
  }

  /**
   * Capture frames with adaptive timing based on video frame rate
   */
  static async captureFramesAdaptive(
    video: HTMLVideoElement,
    options: Omit<MultiFrameCaptureOptions, 'frameDelay'> & {
      /** Target frames per second for capture (default: 10 FPS = 100ms delay) */
      targetFPS?: number;
    } = {}
  ): Promise<MultiFrameCaptureResult> {
    const { targetFPS = 10, ...restOptions } = options;
    const frameDelay = Math.max(50, 1000 / targetFPS); // Minimum 50ms delay
    
    console.log(`🎬 Adaptive capture: targeting ${targetFPS} FPS (${frameDelay}ms delay)`);
    
    return this.captureFrames(video, {
      ...restOptions,
      frameDelay
    });
  }

  /**
   * Capture frames with validation to ensure video is stable
   */
  static async captureFramesWithValidation(
    video: HTMLVideoElement,
    options: MultiFrameCaptureOptions & {
      /** Maximum attempts to get stable frames */
      maxAttempts?: number;
      /** Minimum similarity threshold between consecutive frames (0-1) */
      stabilityThreshold?: number;
    } = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      maxAttempts = 3,
      stabilityThreshold = 0.8,
      ...captureOptions
    } = options;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`🔍 Capture attempt ${attempt}/${maxAttempts}`);
      
      try {
        const result = await this.captureFrames(video, captureOptions);
        
        // Validate frame stability if we have multiple frames
        if (result.frames.length > 1 && stabilityThreshold > 0) {
          const isStable = this.validateFrameStability(result.frames, stabilityThreshold);
          
          if (isStable || attempt === maxAttempts) {
            if (isStable) {
              console.log(`✅ Frames are stable (attempt ${attempt})`);
            } else {
              console.log(`⚠️ Frames not stable but using anyway (final attempt)`);
            }
            return result;
          } else {
            console.log(`❌ Frames not stable, retrying... (attempt ${attempt})`);
            await this.delay(200); // Wait before retry
          }
        } else {
          return result;
        }
      } catch (error) {
        console.error(`❌ Capture attempt ${attempt} failed:`, error);
        if (attempt === maxAttempts) {
          throw error;
        }
        await this.delay(200); // Wait before retry
      }
    }

    throw new Error('Failed to capture stable frames after maximum attempts');
  }

  /**
   * Rapid capture with validation to ensure video is stable
   */
  static async captureFramesRapidWithValidation(
    video: HTMLVideoElement,
    options: Omit<MultiFrameCaptureOptions, 'frameDelay'> & {
      /** Minimum delay between frames in milliseconds (default: 33ms ≈ 30fps) */
      minDelay?: number;
      /** Maximum attempts to get stable frames */
      maxAttempts?: number;
      /** Minimum similarity threshold between consecutive frames (0-1) */
      stabilityThreshold?: number;
    } = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      maxAttempts = 3,
      stabilityThreshold = 0.7, // Slightly more lenient for rapid capture
      minDelay = 33,
      ...captureOptions
    } = options;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`⚡ Rapid capture attempt ${attempt}/${maxAttempts}`);
      
      try {
        const result = await this.captureFramesRapid(video, {
          ...captureOptions,
          minDelay
        });
        
        // Validate frame stability if we have multiple frames
        if (result.frames.length > 1 && stabilityThreshold > 0) {
          const isStable = this.validateFrameStability(result.frames, stabilityThreshold);
          
          if (isStable || attempt === maxAttempts) {
            if (isStable) {
              console.log(`✅ Rapid frames are stable (attempt ${attempt})`);
            } else {
              console.log(`⚠️ Rapid frames not stable but using anyway (final attempt)`);
            }
            return result;
          } else {
            console.log(`❌ Rapid frames not stable, retrying... (attempt ${attempt})`);
            await this.delay(100); // Shorter wait for rapid capture
          }
        } else {
          return result;
        }
      } catch (error) {
        console.error(`❌ Rapid capture attempt ${attempt} failed:`, error);
        if (attempt === maxAttempts) {
          throw error;
        }
        await this.delay(100); // Shorter wait for rapid capture
      }
    }

    throw new Error('Failed to capture stable frames after maximum attempts');
  }

  /**
   * Instant capture with validation to ensure video is stable
   */
  static async captureFramesInstantWithValidation(
    video: HTMLVideoElement,
    options: Omit<MultiFrameCaptureOptions, 'frameDelay'> & {
      /** Maximum attempts to get stable frames */
      maxAttempts?: number;
      /** Minimum similarity threshold between consecutive frames (0-1) */
      stabilityThreshold?: number;
    } = {}
  ): Promise<MultiFrameCaptureResult> {
    const {
      maxAttempts = 3,
      stabilityThreshold = 0.6, // More lenient for instant capture since frames are identical
      ...captureOptions
    } = options;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`🚀 INSTANT capture attempt ${attempt}/${maxAttempts}`);
      
      try {
        const result = await this.captureFramesInstant(video, captureOptions);
        
        // Validate frame stability if we have multiple frames
        if (result.frames.length > 1 && stabilityThreshold > 0) {
          const isStable = this.validateFrameStability(result.frames, stabilityThreshold);
          
          if (isStable || attempt === maxAttempts) {
            if (isStable) {
              console.log(`✅ INSTANT frames are stable (attempt ${attempt})`);
            } else {
              console.log(`⚠️ INSTANT frames not stable but using anyway (final attempt)`);
            }
            return result;
          } else {
            console.log(`❌ INSTANT frames not stable, retrying... (attempt ${attempt})`);
            await this.delay(50); // Very short wait for instant capture
          }
        } else {
          return result;
        }
      } catch (error) {
        console.error(`❌ INSTANT capture attempt ${attempt} failed:`, error);
        if (attempt === maxAttempts) {
          throw error;
        }
        await this.delay(50); // Very short wait for instant capture
      }
    }

    throw new Error('Failed to capture stable frames after maximum attempts');
  }

  /**
   * Validate that captured frames are stable (not too different from each other)
   */
  private static validateFrameStability(
    frames: CapturedFrame[],
    threshold: number
  ): boolean {
    if (frames.length < 2) return true;

    // Compare consecutive frames using a simple pixel difference metric
    for (let i = 1; i < frames.length; i++) {
      const similarity = this.calculateFrameSimilarity(
        frames[i - 1].imageData,
        frames[i].imageData
      );
      
      console.log(`📊 Frame ${i - 1} vs ${i} similarity: ${similarity.toFixed(3)}`);
      
      if (similarity < threshold) {
        return false;
      }
    }

    return true;
  }

  /**
   * Calculate similarity between two frames (0 = completely different, 1 = identical)
   */
  private static calculateFrameSimilarity(
    frame1: ImageData,
    frame2: ImageData
  ): number {
    if (frame1.width !== frame2.width || frame1.height !== frame2.height) {
      return 0;
    }

    const data1 = frame1.data;
    const data2 = frame2.data;
    let totalDiff = 0;
    let pixelCount = 0;

    // Sample every 4th pixel for performance (RGBA = 4 bytes per pixel)
    for (let i = 0; i < data1.length; i += 16) { // Skip 4 pixels at a time
      const r1 = data1[i];
      const g1 = data1[i + 1];
      const b1 = data1[i + 2];
      
      const r2 = data2[i];
      const g2 = data2[i + 1];
      const b2 = data2[i + 2];
      
      // Calculate Euclidean distance in RGB space
      const diff = Math.sqrt(
        Math.pow(r1 - r2, 2) +
        Math.pow(g1 - g2, 2) +
        Math.pow(b1 - b2, 2)
      );
      
      totalDiff += diff;
      pixelCount++;
    }

    const avgDiff = totalDiff / pixelCount;
    const maxPossibleDiff = Math.sqrt(3 * Math.pow(255, 2)); // Max RGB distance
    const similarity = 1 - (avgDiff / maxPossibleDiff);
    
    return Math.max(0, Math.min(1, similarity));
  }

  /**
   * Utility function for creating delays
   */
  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get the middle frame from a capture result
   */
  static getMiddleFrame(result: MultiFrameCaptureResult): CapturedFrame {
    return result.frames[result.middleFrameIndex];
  }

  /**
   * Get all frame ImageData for batch inference
   */
  static getFrameImageData(result: MultiFrameCaptureResult): ImageData[] {
    return result.frames.map(frame => frame.imageData);
  }
}
