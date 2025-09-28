import { BackboneModelConfig } from '../utils/BackboneModelManager';

/**
 * Configuration for the YOLO OBB backbone model with JavaScript NMS
 */
export const BACKBONE_MODEL_CONFIG: BackboneModelConfig = {
  modelPath: '/models/trading_card_detector_backbone.onnx',
  inputSize: 1088, // Model input size from model_info.json
  executionProviders: [
    'webgl', // Try WebGL first with ONNX Runtime 1.23.0
    'wasm'   // Fallback to WASM
  ],
  nmsConfig: {
    confidenceThreshold: 0.8,  // High threshold to match validation results (0.82+)
    nmsThreshold: 0.4,         // Stricter NMS threshold
    maxDetections: 3,          // Very few detections for clean results
    inputSize: 1088            // Must match model input size
  }
};

/**
 * Alternative configuration for CPU-only execution
 */
export const CPU_MODEL_CONFIG: BackboneModelConfig = {
  ...BACKBONE_MODEL_CONFIG,
  executionProviders: ['wasm'], // CPU only
  nmsConfig: {
    ...BACKBONE_MODEL_CONFIG.nmsConfig,
    confidenceThreshold: 0.8, // High threshold to match validation
    nmsThreshold: 0.4,        // Stricter NMS for clean results
    maxDetections: 3,         // Fewer detections for performance
    inputSize: 1088            // Must match model input size
  }
};

/**
 * High-performance configuration for powerful devices
 */
export const HIGH_PERFORMANCE_CONFIG: BackboneModelConfig = {
  ...BACKBONE_MODEL_CONFIG,
  nmsConfig: {
    ...BACKBONE_MODEL_CONFIG.nmsConfig,
    confidenceThreshold: 0.8, // High threshold to match validation
    nmsThreshold: 0.4,        // Stricter NMS for cleaner results
    maxDetections: 3,         // Fewer detections for clean results
    inputSize: 1088            // Must match model input size
  }
};

/**
 * Mobile-optimized configuration
 */
export const MOBILE_CONFIG: BackboneModelConfig = {
  ...BACKBONE_MODEL_CONFIG,
  executionProviders: ['wasm'], // CPU only for mobile compatibility
  nmsConfig: {
    ...BACKBONE_MODEL_CONFIG.nmsConfig,
    confidenceThreshold: 0.8, // High threshold for performance
    nmsThreshold: 0.4,        // Stricter NMS for clean results
    maxDetections: 3,         // Fewer detections for mobile
    inputSize: 1088            // Must match model input size
  }
};

/**
 * Get model configuration based on device capabilities
 */
export function getOptimalModelConfig(): BackboneModelConfig {
  // Detect device capabilities
  const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const hasWebGL = !!document.createElement('canvas').getContext('webgl2');
  const hasHighMemory = (navigator as any).deviceMemory ? (navigator as any).deviceMemory >= 4 : true;

  console.log('🔍 Device detection:', {
    isMobile,
    hasWebGL,
    hasHighMemory,
    userAgent: navigator.userAgent
  });

  if (isMobile) {
    console.log('📱 Using mobile-optimized configuration');
    return MOBILE_CONFIG;
  }

  if (hasWebGL && hasHighMemory) {
    console.log('🚀 Using high-performance configuration');
    return HIGH_PERFORMANCE_CONFIG;
  }

  if (hasWebGL) {
    console.log('⚡ Using standard WebGL configuration');
    return BACKBONE_MODEL_CONFIG;
  }

  console.log('🔧 Using CPU-only configuration');
  return CPU_MODEL_CONFIG;
}
