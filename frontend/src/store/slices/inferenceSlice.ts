import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface CardDetection {
  id: string;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  cardName?: string;
  setName?: string;
  corners?: Array<{ x: number; y: number }>; // 4 corner points for rotated bounding box
  isRotated?: boolean;
}

export interface InferenceResult {
  detections: CardDetection[];
  processingTime: number;
  timestamp: number;
  imageWidth: number;
  imageHeight: number;
}

export interface ModelState {
  isLoaded: boolean;
  isLoading: boolean;
  modelPath: string | null;
  version: string | null;
  error: string | null;
}

interface InferenceSliceState {
  model: ModelState;
  isProcessing: boolean;
  lastResult: InferenceResult | null;
  processingQueue: number;
  averageProcessingTime: number;
  totalInferences: number;
  detectionHistory: InferenceResult[];
  maxHistorySize: number;
}

const initialState: InferenceSliceState = {
  model: {
    isLoaded: false,
    isLoading: false,
    modelPath: null,
    version: null,
    error: null,
  },
  isProcessing: false,
  lastResult: null,
  processingQueue: 0,
  averageProcessingTime: 0,
  totalInferences: 0,
  detectionHistory: [],
  maxHistorySize: 10,
};

const inferenceSlice = createSlice({
  name: 'inference',
  initialState,
  reducers: {
    setModelLoading: (state, action: PayloadAction<boolean>) => {
      state.model.isLoading = action.payload;
      if (action.payload) {
        state.model.error = null;
      }
    },
    
    setModelLoaded: (state, action: PayloadAction<{ path: string; version: string }>) => {
      state.model.isLoaded = true;
      state.model.isLoading = false;
      state.model.modelPath = action.payload.path;
      state.model.version = action.payload.version;
      state.model.error = null;
    },
    
    setModelError: (state, action: PayloadAction<string>) => {
      state.model.error = action.payload;
      state.model.isLoading = false;
      state.model.isLoaded = false;
    },
    
    setProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    
    addToQueue: (state) => {
      state.processingQueue += 1;
    },
    
    removeFromQueue: (state) => {
      state.processingQueue = Math.max(0, state.processingQueue - 1);
    },
    
    setInferenceResult: (state, action: PayloadAction<InferenceResult>) => {
      const result = action.payload;
      state.lastResult = result;
      
      // Update processing statistics
      state.totalInferences += 1;
      const newAverage = (state.averageProcessingTime * (state.totalInferences - 1) + result.processingTime) / state.totalInferences;
      state.averageProcessingTime = newAverage;
      
      // Add to history
      state.detectionHistory.unshift(result);
      if (state.detectionHistory.length > state.maxHistorySize) {
        state.detectionHistory = state.detectionHistory.slice(0, state.maxHistorySize);
      }
    },
    
    clearResults: (state) => {
      state.lastResult = null;
      state.detectionHistory = [];
    },
    
    resetInference: (state) => {
      return {
        ...initialState,
        model: state.model, // Keep model state
      };
    },
  },
});

export const {
  setModelLoading,
  setModelLoaded,
  setModelError,
  setProcessing,
  addToQueue,
  removeFromQueue,
  setInferenceResult,
  clearResults,
  resetInference,
} = inferenceSlice.actions;

export default inferenceSlice.reducer;
