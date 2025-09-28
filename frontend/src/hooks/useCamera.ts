import { useEffect, useRef, useCallback } from 'react';
import { CameraError } from '../types/camera';
import { AdaptiveCameraSystem } from '../utils/AdaptiveCameraSystem';
import { useAppDispatch, useAppSelector } from './redux';
import {
  setStatus,
  setInitialized,
  setStreaming,
  setStream,
  setError,
  clearError,
  setPerformanceMode,
  setCapturing,
  setBatteryLevel,
} from '../store/slices/cameraSlice';

export const useCamera = () => {
  const dispatch = useAppDispatch();
  const cameraState = useAppSelector(state => state.camera);
  const cameraSystemRef = useRef<AdaptiveCameraSystem | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Initialize camera system
  useEffect(() => {
    cameraSystemRef.current = new AdaptiveCameraSystem();
    dispatch(setPerformanceMode(cameraSystemRef.current.getPerformanceMode()));
  }, [dispatch]);

  const initializeCamera = useCallback(async () => {
    if (!cameraSystemRef.current) return;

    dispatch(setStatus('initializing'));
    dispatch(clearError());

    try {
      const stream = await cameraSystemRef.current.initializeCamera();
      
      dispatch(setInitialized(true));
      dispatch(setStreaming(true));
      dispatch(setStream(stream));
      dispatch(setStatus('streaming'));

      // Attach stream to video element if available
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

    } catch (error) {
      const cameraError = error as CameraError;
      dispatch(setError(cameraError));
      dispatch(setInitialized(false));
      dispatch(setStreaming(false));
    }
  }, [dispatch]);

  const retryWithFallback = useCallback(async () => {
    if (!cameraSystemRef.current) return;

    dispatch(setStatus('initializing'));
    
    try {
      const stream = await cameraSystemRef.current.retryWithFallbackConstraints();
      
      dispatch(setInitialized(true));
      dispatch(setStreaming(true));
      dispatch(setStream(stream));
      dispatch(clearError());
      dispatch(setStatus('streaming'));

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

    } catch (error) {
      const cameraError = error as CameraError;
      dispatch(setError(cameraError));
    }
  }, [dispatch]);

  const stopCamera = useCallback(() => {
    if (cameraSystemRef.current) {
      cameraSystemRef.current.stopCamera();
    }

    dispatch(setStreaming(false));
    dispatch(setStream(null));
    dispatch(setStatus('idle'));

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [dispatch]);

  const attachVideoElement = useCallback((video: HTMLVideoElement) => {
    videoRef.current = video;
    if (cameraState.stream) {
      video.srcObject = cameraState.stream;
    }
  }, [cameraState.stream]);

  const shouldSkipFrame = useCallback((): boolean => {
    return cameraSystemRef.current?.shouldSkipFrame() || false;
  }, []);

  const adaptForBattery = useCallback(async () => {
    if (cameraSystemRef.current) {
      await cameraSystemRef.current.adaptFrameRateForBattery();
      const batteryLevel = await cameraSystemRef.current.getBatteryLevel();
      dispatch(setBatteryLevel(batteryLevel));
    }
  }, [dispatch]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cameraSystemRef.current) {
        cameraSystemRef.current.stopCamera();
      }
    };
  }, []);

  return {
    cameraState,
    initializeCamera,
    retryWithFallback,
    stopCamera,
    attachVideoElement,
    shouldSkipFrame,
    adaptForBattery,
    cameraSystem: cameraSystemRef.current
  };
};
