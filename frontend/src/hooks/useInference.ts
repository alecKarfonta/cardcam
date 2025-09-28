import { useCallback, useEffect, useRef } from 'react';
import { useAppDispatch, useAppSelector } from './redux';
import {
  setModelLoading,
  setModelLoaded,
  setModelError,
  setProcessing,
  addToQueue,
  removeFromQueue,
  setInferenceResult,
  InferenceResult,
  CardDetection,
} from '../store/slices/inferenceSlice';
import { ModelManager, ModelConfig, ModelPrediction } from '../utils/ModelManager';
import { PerformanceMonitor } from '../utils/PerformanceMonitor';

const DEFAULT_MODEL_CONFIG: ModelConfig = {
  modelPath: `${process.env.PUBLIC_URL || ''}/models/trading_card_detector.onnx`,
  inputShape: [1, 3, 1088, 1088],
  outputShape: [1, 300, 7], // NMS-enabled ONNX format: [cx,cy,w,h,conf,class,angle] per detection
  executionProviders: ['webgl', 'wasm', 'cpu'], // Try WebGL first for better performance
  confidenceThreshold: 0.25, // Lower threshold since NMS is already applied
  nmsThreshold: 0.45, // Not used since NMS is built into ONNX model
};

export const useInference = () => {
  const dispatch = useAppDispatch();
  const inferenceState = useAppSelector((state: any) => state.inference);
  const modelManagerRef = useRef<ModelManager | null>(null);
  const performanceMonitorRef = useRef<PerformanceMonitor | null>(null);

  // Initialize model manager and performance monitor
  useEffect(() => {
    modelManagerRef.current = new ModelManager(DEFAULT_MODEL_CONFIG);
    performanceMonitorRef.current = new PerformanceMonitor();
    
    // Make performance monitor available globally for debugging
    (window as any).performanceMonitor = performanceMonitorRef.current;
    console.log('🔧 Performance monitor available at window.performanceMonitor');
  }, []);

  const loadModel = useCallback(async (modelPath?: string) => {
    if (!modelManagerRef.current) {
      console.error('❌ ModelManager not initialized');
      return;
    }

    console.log('🔄 Starting model load process...');
    dispatch(setModelLoading(true));

    try {
      if (modelPath) {
        console.log('🔧 Updating model path to:', modelPath);
        modelManagerRef.current.updateConfig({ modelPath });
      }

      console.log('📦 Loading model...');
      await modelManagerRef.current.loadModel();
      
      const config = modelManagerRef.current.getConfig();
      console.log('✅ Model loaded successfully! Dispatching success action...');
      dispatch(setModelLoaded({
        path: config.modelPath,
        version: '1.0.0' // TODO: Get from model metadata
      }));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ Model loading failed:', errorMessage);
      dispatch(setModelError(errorMessage));
    }
  }, [dispatch]);

  const runInference = useCallback(async (
    imageData: ImageData,
    options?: {
      skipQueue?: boolean;
      callback?: (result: InferenceResult) => void;
    }
  ): Promise<InferenceResult | null> => {
    if (!modelManagerRef.current || !modelManagerRef.current.isModelLoaded()) {
      console.warn('Model not loaded, skipping inference');
      return null;
    }

    if (!options?.skipQueue) {
      dispatch(addToQueue());
    }

    dispatch(setProcessing(true));

    try {
      const startTime = performance.now();
      const prediction = await modelManagerRef.current.predict(imageData);
      const processingTime = performance.now() - startTime;

      // Record performance metrics
      if (performanceMonitorRef.current && modelManagerRef.current) {
        const config = modelManagerRef.current.getConfig();
        performanceMonitorRef.current.recordMetrics({
          inferenceTime: processingTime,
          preprocessingTime: 0, // Not separately tracked in current implementation
          postprocessingTime: 0, // Not separately tracked in current implementation
          totalTime: processingTime,
          executionProvider: config.executionProviders[0] || 'unknown',
          modelPath: config.modelPath,
          inputSize: { width: imageData.width, height: imageData.height },
          timestamp: Date.now()
        });
      }

      // Convert model prediction to inference result
      const result: InferenceResult = {
        detections: convertPredictionToDetections(prediction, imageData.width, imageData.height),
        processingTime,
        timestamp: Date.now(),
        imageWidth: imageData.width,
        imageHeight: imageData.height,
      };

      console.log(`🎯 Inference result: ${result.detections.length} detections found in ${processingTime.toFixed(2)}ms`);
      if (result.detections.length > 0) {
        console.log('📊 Detection details:', result.detections.map(d => ({
          confidence: d.confidence.toFixed(3),
          boundingBox: d.boundingBox,
          isRotated: d.isRotated,
          corners: d.corners
        })));
      }

      dispatch(setInferenceResult(result));

      if (options?.callback) {
        options.callback(result);
      }

      return result;

    } catch (error) {
      console.error('Inference failed:', error);
      return null;
    } finally {
      dispatch(setProcessing(false));
      if (!options?.skipQueue) {
        dispatch(removeFromQueue());
      }
    }
  }, [dispatch]);

  const runBatchInference = useCallback(async (
    imageDataArray: ImageData[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<InferenceResult[]> => {
    const results: InferenceResult[] = [];

    for (let i = 0; i < imageDataArray.length; i++) {
      const result = await runInference(imageDataArray[i], { skipQueue: true });
      if (result) {
        results.push(result);
      }

      if (onProgress) {
        onProgress(i + 1, imageDataArray.length);
      }
    }

    return results;
  }, [runInference]);

  const updateModelConfig = useCallback((config: Partial<ModelConfig>) => {
    if (modelManagerRef.current) {
      modelManagerRef.current.updateConfig(config);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (modelManagerRef.current) {
        modelManagerRef.current.dispose();
      }
    };
  }, []);

  return {
    modelState: inferenceState.model || { isLoaded: false, isLoading: false, modelPath: null, version: null, error: null },
    isProcessing: inferenceState.isProcessing || false,
    lastResult: inferenceState.lastResult || null,
    processingQueue: inferenceState.processingQueue || 0,
    averageProcessingTime: inferenceState.averageProcessingTime || 0,
    detectionHistory: inferenceState.detectionHistory || [],
    loadModel,
    runInference,
    runBatchInference,
    updateModelConfig,
    modelManager: modelManagerRef.current, // Expose ModelManager for PostProcessValidator
    performanceMonitor: performanceMonitorRef.current, // Expose PerformanceMonitor
  };
};

// Helper function to convert model predictions to card detections
function convertPredictionToDetections(
  prediction: ModelPrediction,
  imageWidth: number,
  imageHeight: number
): CardDetection[] {
  const detections: CardDetection[] = [];

  for (let i = 0; i < prediction.validDetections; i++) {
    const boxIndex = i * 4;
    
    // Convert normalized coordinates to pixel coordinates for axis-aligned box
    const x = prediction.boxes[boxIndex] * imageWidth;
    const y = prediction.boxes[boxIndex + 1] * imageHeight;
    const width = prediction.boxes[boxIndex + 2] * imageWidth;
    const height = prediction.boxes[boxIndex + 3] * imageHeight;

    const detection: CardDetection = {
      id: `detection_${Date.now()}_${i}`,
      boundingBox: {
        x: Math.max(0, x),
        y: Math.max(0, y),
        width: Math.min(width, imageWidth - x),
        height: Math.min(height, imageHeight - y),
      },
      confidence: prediction.scores[i],
      isRotated: !!prediction.rotatedBoxes,
    };

    // Add rotated bounding box corners if available
    if (prediction.rotatedBoxes && prediction.rotatedBoxes.length > i * 8) {
      const rotatedIndex = i * 8;
      detection.corners = [
        {
          x: prediction.rotatedBoxes[rotatedIndex] * imageWidth,
          y: prediction.rotatedBoxes[rotatedIndex + 1] * imageHeight,
        },
        {
          x: prediction.rotatedBoxes[rotatedIndex + 2] * imageWidth,
          y: prediction.rotatedBoxes[rotatedIndex + 3] * imageHeight,
        },
        {
          x: prediction.rotatedBoxes[rotatedIndex + 4] * imageWidth,
          y: prediction.rotatedBoxes[rotatedIndex + 5] * imageHeight,
        },
        {
          x: prediction.rotatedBoxes[rotatedIndex + 6] * imageWidth,
          y: prediction.rotatedBoxes[rotatedIndex + 7] * imageHeight,
        },
      ];
    }

    detections.push(detection);
  }

  return detections;
}
