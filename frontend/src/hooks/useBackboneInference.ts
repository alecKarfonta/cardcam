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
import { BackboneModelManager, BackboneModelConfig, BackboneModelPrediction } from '../utils/BackboneModelManager';
import { OBBDetection } from '../utils/OBBNMSProcessor';
import { getOptimalModelConfig } from '../config/modelConfig';

/**
 * Hook for using the YOLO OBB backbone model with JavaScript NMS post-processing
 * This replaces the old useInference hook with the new pipeline approach
 */
export const useBackboneInference = () => {
  const dispatch = useAppDispatch();
  const inferenceState = useAppSelector((state: any) => state.inference);
  const modelManagerRef = useRef<BackboneModelManager | null>(null);

  // Initialize backbone model manager with optimal configuration
  useEffect(() => {
    const config = getOptimalModelConfig();
    console.log('🔧 Initializing backbone model manager with config:', config);
    modelManagerRef.current = new BackboneModelManager(config);
  }, []);

  const loadModel = useCallback(async (customConfig?: Partial<BackboneModelConfig>) => {
    if (!modelManagerRef.current) {
      console.error('❌ BackboneModelManager not initialized');
      return;
    }

    console.log('🔄 Starting backbone model load process...');
    dispatch(setModelLoading(true));

    try {
      if (customConfig) {
        console.log('🔧 Updating model configuration:', customConfig);
        modelManagerRef.current.updateConfig(customConfig);
      }

      console.log('📦 Loading backbone model...');
      await modelManagerRef.current.loadModel();
      
      const config = modelManagerRef.current.getConfig();
      console.log('✅ Backbone model loaded successfully!');
      console.log('🎯 Model configuration:', {
        modelPath: config.modelPath,
        inputSize: config.inputSize,
        executionProviders: config.executionProviders,
        nmsConfig: config.nmsConfig
      });

      dispatch(setModelLoaded({
        path: config.modelPath,
        version: '2.0.0-backbone' // Version identifier for backbone model
      }));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ Backbone model loading failed:', errorMessage);
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
      console.warn('⚠️ Backbone model not loaded, skipping inference');
      return null;
    }

    if (!options?.skipQueue) {
      dispatch(addToQueue());
    }

    dispatch(setProcessing(true));

    try {
      const startTime = performance.now();
      console.log(`🚀 Running backbone inference on ${imageData.width}x${imageData.height} image...`);
      
      const prediction = await modelManagerRef.current.predict(imageData);
      const totalTime = performance.now() - startTime;

      console.log(`⚡ Complete pipeline finished in ${totalTime.toFixed(2)}ms`);
      console.log(`📊 Pipeline breakdown: inference=${prediction.inferenceTime.toFixed(2)}ms, processing=${prediction.processingTime.toFixed(2)}ms`);

      // Convert backbone prediction to inference result
      // Note: We keep coordinates normalized (0-1) and let the overlay handle canvas scaling
      const result: InferenceResult = {
        detections: convertOBBDetectionsToCardDetections(prediction.detections, imageData.width, imageData.height),
        processingTime: totalTime,
        timestamp: Date.now(),
        imageWidth: imageData.width,
        imageHeight: imageData.height,
      };

      console.log(`🎯 Final result: ${result.detections.length} detections found`);
      if (result.detections.length > 0) {
        console.log('📊 Detection details:', result.detections.map(d => ({
          confidence: d.confidence.toFixed(3),
          boundingBox: d.boundingBox,
          isRotated: d.isRotated,
          corners: d.corners?.length
        })));
      }

      dispatch(setInferenceResult(result));

      if (options?.callback) {
        options.callback(result);
      }

      return result;

    } catch (error) {
      console.error('❌ Backbone inference failed:', error);
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
    console.log(`🔄 Running batch inference on ${imageDataArray.length} images...`);
    const results: InferenceResult[] = [];

    for (let i = 0; i < imageDataArray.length; i++) {
      console.log(`📊 Processing image ${i + 1}/${imageDataArray.length}...`);
      
      // Yield to browser between each inference to prevent UI blocking
      if (i > 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      const result = await runInference(imageDataArray[i], { skipQueue: true });
      if (result) {
        results.push(result);
      }

      if (onProgress) {
        onProgress(i + 1, imageDataArray.length);
      }
    }

    console.log(`✅ Batch inference complete: ${results.length}/${imageDataArray.length} successful`);
    return results;
  }, [runInference]);

  const updateModelConfig = useCallback((config: Partial<BackboneModelConfig>) => {
    if (modelManagerRef.current) {
      console.log('🔧 Updating backbone model configuration:', config);
      modelManagerRef.current.updateConfig(config);
    }
  }, []);

  const updateNMSConfig = useCallback((nmsConfig: Partial<BackboneModelConfig['nmsConfig']>) => {
    if (modelManagerRef.current) {
      console.log('🔧 Updating NMS configuration:', nmsConfig);
      modelManagerRef.current.updateNMSConfig(nmsConfig);
    }
  }, []);

  const getCurrentConfig = useCallback(() => {
    return modelManagerRef.current?.getConfig() || null;
  }, []);

  const getNMSConfig = useCallback(() => {
    return modelManagerRef.current?.getNMSConfig() || null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (modelManagerRef.current) {
        console.log('🗑️ Cleaning up backbone model manager...');
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
    backboneModelManager: modelManagerRef.current, // Expose BackboneModelManager for PostProcessValidator
    updateNMSConfig,
    getCurrentConfig,
    getNMSConfig,
  };
};

/**
 * Convert OBB detections from backbone model to CardDetection format
 * CRITICAL: DetectionOverlay expects coordinates in canvas space, not image space
 */
function convertOBBDetectionsToCardDetections(
  obbDetections: OBBDetection[],
  imageWidth: number,
  imageHeight: number
): CardDetection[] {
  console.log(`🔄 Converting ${obbDetections.length} OBB detections for canvas display`);
  console.log(`📐 Image dimensions: ${imageWidth}x${imageHeight}`);
  
  return obbDetections.map((detection, index) => {
    // Debug the raw detection data
    console.log(`🔍 Detection ${index}:`, {
      boundingBox: detection.boundingBox,
      corners: detection.corners,
      confidence: detection.confidence
    });

    // IMPORTANT: Keep coordinates normalized (0-1) for now
    // The DetectionOverlay will scale them to canvas dimensions
    const boundingBox = {
      x: detection.boundingBox.x,      // Keep normalized
      y: detection.boundingBox.y,      // Keep normalized
      width: detection.boundingBox.width,   // Keep normalized
      height: detection.boundingBox.height, // Keep normalized
    };

    // Keep corner coordinates normalized too
    const corners = [
      {
        x: detection.corners.x1,  // Keep normalized
        y: detection.corners.y1,  // Keep normalized
      },
      {
        x: detection.corners.x2,  // Keep normalized
        y: detection.corners.y2,  // Keep normalized
      },
      {
        x: detection.corners.x3,  // Keep normalized
        y: detection.corners.y3,  // Keep normalized
      },
      {
        x: detection.corners.x4,  // Keep normalized
        y: detection.corners.y4,  // Keep normalized
      },
    ];

    const cardDetection: CardDetection = {
      id: `backbone_detection_${Date.now()}_${index}`,
      boundingBox,
      confidence: detection.confidence,
      isRotated: true, // Always true for OBB detections
      corners,
    };

    console.log(`🔍 Detection ${index}:`, {
      boundingBox: cardDetection.boundingBox,
      corners: cardDetection.corners?.map((c, i) => `C${i}:(${c.x.toFixed(3)},${c.y.toFixed(3)})`).join(' '),
      confidence: cardDetection.confidence,
      isRotated: cardDetection.isRotated,
      angle: detection.angle
    });

    return cardDetection;
  });
}
