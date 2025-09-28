import { CameraConstraints, CameraError, DeviceCapability, CameraConfig } from '../types/camera';

export class AdaptiveCameraSystem {
  private stream: MediaStream | null = null;
  private performanceMode: 'low' | 'medium' | 'high';
  private config: CameraConfig;
  private processingEnabled: boolean = true;
  private frameSkipCounter: number = 0;
  private lastFrameTime: number = 0;

  constructor() {
    this.performanceMode = this.detectDeviceCapability();
    this.config = this.generateConfig();
  }

  private detectDeviceCapability(): 'low' | 'medium' | 'high' {
    const cores = navigator.hardwareConcurrency || 2;
    const memory = (navigator as any).deviceMemory || 4;
    
    if (cores >= 8 && memory >= 8) return 'high';
    if (cores >= 4 && memory >= 4) return 'medium';
    return 'low';
  }

  private generateConfig(): CameraConfig {
    const configs = {
      low: { targetFPS: 15, frameSkipThreshold: 4 },
      medium: { targetFPS: 24, frameSkipThreshold: 2 },
      high: { targetFPS: 30, frameSkipThreshold: 1 }
    };

    const config = configs[this.performanceMode];
    return {
      ...config,
      processingEnabled: true,
      adaptiveQuality: true
    };
  }

  async initializeCamera(): Promise<MediaStream> {
    try {
      // Check permissions first
      await this.checkCameraPermissions();

      // Mobile-optimized constraints for card scanning
      const constraints: CameraConstraints = {
        video: {
          facingMode: { ideal: 'environment' }, // Rear camera preferred
          width: { min: 1280, ideal: 1920, max: 3840 },
          height: { min: 720, ideal: 1080, max: 2160 },
          frameRate: { ideal: this.config.targetFPS, max: 60 }
        },
        audio: false
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      return this.stream;
    } catch (error) {
      throw this.handleCameraError(error as DOMException);
    }
  }

  private async checkCameraPermissions(): Promise<void> {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new DOMException('Camera API not supported', 'NotSupportedError');
    }

    try {
      const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
      if (permissions.state === 'denied') {
        throw new DOMException('Camera permission denied', 'NotAllowedError');
      }
    } catch (error) {
      // Permissions API might not be supported, continue with getUserMedia
      console.warn('Permissions API not supported, proceeding with getUserMedia');
    }
  }

  private handleCameraError(error: DOMException): CameraError {
    const errorMap: Record<string, CameraError> = {
      'NotAllowedError': {
        name: 'NotAllowedError',
        message: 'Camera permission denied. Please allow camera access to scan trading cards.'
      },
      'NotFoundError': {
        name: 'NotFoundError',
        message: 'No camera found. Please ensure your device has a camera.'
      },
      'OverconstrainedError': {
        name: 'OverconstrainedError',
        message: 'Camera constraints not supported. Trying with lower quality settings...'
      },
      'NotSupportedError': {
        name: 'NotSupportedError',
        message: 'Camera API not supported on this device.'
      }
    };

    return errorMap[error.name] || {
      name: 'UnknownError',
      message: 'An unknown camera error occurred. Please refresh and try again.'
    };
  }

  async retryWithFallbackConstraints(): Promise<MediaStream> {
    const fallbackConstraints: CameraConstraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
      return this.stream;
    } catch (error) {
      throw this.handleCameraError(error as DOMException);
    }
  }

  shouldSkipFrame(): boolean {
    if (!this.config.adaptiveQuality) return false;

    const now = performance.now();
    const timeSinceLastFrame = now - this.lastFrameTime;
    const targetFrameTime = 1000 / this.config.targetFPS;

    // Skip frame if we're processing too fast
    if (timeSinceLastFrame < targetFrameTime) return true;

    // Adaptive frame skipping based on performance
    this.frameSkipCounter++;
    const shouldSkip = this.frameSkipCounter % this.config.frameSkipThreshold !== 0;
    
    if (!shouldSkip) {
      this.lastFrameTime = now;
    }

    return shouldSkip;
  }

  async getBatteryLevel(): Promise<number> {
    try {
      const battery = await (navigator as any).getBattery();
      return battery.level;
    } catch (error) {
      return 1; // Assume full battery if API not available
    }
  }

  async adaptFrameRateForBattery(): Promise<void> {
    const batteryLevel = await this.getBatteryLevel();
    
    if (batteryLevel < 0.2) {
      this.config.frameSkipThreshold = 6; // Very conservative
      this.config.targetFPS = 10;
    } else if (batteryLevel < 0.5) {
      this.config.frameSkipThreshold = 3; // Moderate conservation
      this.config.targetFPS = 20;
    } else {
      // Reset to performance-based settings
      this.config = this.generateConfig();
    }
  }

  getPerformanceMode(): 'low' | 'medium' | 'high' {
    return this.performanceMode;
  }

  getConfig(): CameraConfig {
    return { ...this.config };
  }

  stopCamera(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  isStreaming(): boolean {
    return this.stream !== null && this.stream.active;
  }
}
