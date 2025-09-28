export interface CameraConstraints {
  video: {
    facingMode?: { ideal: string };
    width?: { min?: number; ideal?: number; max?: number };
    height?: { min?: number; ideal?: number; max?: number };
    frameRate?: { ideal?: number; max?: number };
  };
  audio: boolean;
}

export interface CameraError {
  name: string;
  message: string;
  constraint?: string;
}

export interface CameraState {
  isInitialized: boolean;
  isStreaming: boolean;
  error: CameraError | null;
  stream: MediaStream | null;
  deviceId: string | null;
  facingMode: 'user' | 'environment';
  performanceMode: 'low' | 'medium' | 'high';
}

export interface DeviceCapability {
  hardwareConcurrency: number;
  memory?: number;
  batteryLevel?: number;
  performanceMode: 'low' | 'medium' | 'high';
}

export interface CameraPermissionState {
  granted: boolean;
  denied: boolean;
  prompt: boolean;
}

export type CameraStatus = 'idle' | 'initializing' | 'streaming' | 'error' | 'processing';

export interface CameraConfig {
  targetFPS: number;
  processingEnabled: boolean;
  adaptiveQuality: boolean;
  frameSkipThreshold: number;
}
