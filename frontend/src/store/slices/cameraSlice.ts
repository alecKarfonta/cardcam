import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { CameraState, CameraError, CameraStatus } from '../../types/camera';

interface CameraSliceState extends CameraState {
  status: CameraStatus;
  isCapturing: boolean;
  lastCaptureTime: number | null;
  frameRate: number;
  batteryLevel: number;
}

const initialState: CameraSliceState = {
  isInitialized: false,
  isStreaming: false,
  error: null,
  stream: null,
  deviceId: null,
  facingMode: 'environment',
  performanceMode: 'medium',
  status: 'idle',
  isCapturing: false,
  lastCaptureTime: null,
  frameRate: 30,
  batteryLevel: 1,
};

const cameraSlice = createSlice({
  name: 'camera',
  initialState,
  reducers: {
    setStatus: (state, action: PayloadAction<CameraStatus>) => {
      state.status = action.payload;
    },
    
    setInitialized: (state, action: PayloadAction<boolean>) => {
      state.isInitialized = action.payload;
    },
    
    setStreaming: (state, action: PayloadAction<boolean>) => {
      state.isStreaming = action.payload;
      if (!action.payload) {
        state.stream = null;
      }
    },
    
    setStream: (state, action: PayloadAction<MediaStream | null>) => {
      state.stream = action.payload;
      state.isStreaming = action.payload !== null;
    },
    
    setError: (state, action: PayloadAction<CameraError | null>) => {
      state.error = action.payload;
      if (action.payload) {
        state.status = 'error';
        state.isStreaming = false;
        state.stream = null;
      }
    },
    
    clearError: (state) => {
      state.error = null;
      if (state.status === 'error') {
        state.status = 'idle';
      }
    },
    
    setFacingMode: (state, action: PayloadAction<'user' | 'environment'>) => {
      state.facingMode = action.payload;
    },
    
    setPerformanceMode: (state, action: PayloadAction<'low' | 'medium' | 'high'>) => {
      state.performanceMode = action.payload;
    },
    
    setCapturing: (state, action: PayloadAction<boolean>) => {
      state.isCapturing = action.payload;
      if (action.payload) {
        state.lastCaptureTime = Date.now();
      }
    },
    
    setFrameRate: (state, action: PayloadAction<number>) => {
      state.frameRate = action.payload;
    },
    
    setBatteryLevel: (state, action: PayloadAction<number>) => {
      state.batteryLevel = action.payload;
    },
    
    resetCamera: (state) => {
      return {
        ...initialState,
        performanceMode: state.performanceMode, // Keep performance mode
      };
    },
  },
});

export const {
  setStatus,
  setInitialized,
  setStreaming,
  setStream,
  setError,
  clearError,
  setFacingMode,
  setPerformanceMode,
  setCapturing,
  setFrameRate,
  setBatteryLevel,
  resetCamera,
} = cameraSlice.actions;

export default cameraSlice.reducer;
