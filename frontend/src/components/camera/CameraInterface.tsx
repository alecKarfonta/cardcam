import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import { useCamera } from '../../hooks/useCamera';
import { useInference } from '../../hooks/useInference';
import { useAppDispatch } from '../../hooks/redux';
import { DetectionOverlay } from './DetectionOverlay';
import { CardDetectionSystem } from '../../utils/CardDetectionSystem';
import { CardCropper } from '../../utils/CardCropper';
import { setCurrentView } from '../../store/slices/appSlice';
import { 
  startExtraction, 
  addExtractedCard, 
  completeExtraction, 
  updateExtractionProgress 
} from '../../store/slices/cardExtractionSlice';
import './CameraInterface.css';

interface CameraInterfaceProps {
  onCapture?: (imageData: ImageData) => void;
  onError?: (error: string) => void;
}

export interface CameraInterfaceRef {
  resumeCamera: () => void;
}

export const CameraInterface = forwardRef<CameraInterfaceRef, CameraInterfaceProps>(({
  onCapture,
  onError
}, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionSystemRef = useRef<CardDetectionSystem>(new CardDetectionSystem());
  const [detectionState, setDetectionState] = useState<'searching' | 'detected' | 'positioned' | 'ready'>('searching');
  const [isProcessing, setIsProcessing] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.8);
  const [isCameraFrozen, setIsCameraFrozen] = useState(false);
  const [frozenFrameData, setFrozenFrameData] = useState<ImageData | null>(null);
  const modelLoadedRef = useRef<boolean>(false);
  const frozenCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const dispatch = useAppDispatch();

  const {
    cameraState,
    initializeCamera,
    retryWithFallback,
    stopCamera,
    attachVideoElement,
    shouldSkipFrame
  } = useCamera();

  const {
    modelState,
    isProcessing: isInferenceProcessing,
    lastResult,
    loadModel,
    runInference
  } = useInference();

  // Capture current frame with detections for debugging
  const captureDebugFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) {
      console.warn('⚠️ Cannot capture frame: missing video or canvas');
      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

    // Create raw frame canvas
    const rawCanvas = document.createElement('canvas');
    const rawCtx = rawCanvas.getContext('2d');
    if (!rawCtx) return;

    // Set canvas size to match video
    rawCanvas.width = video.videoWidth;
    rawCanvas.height = video.videoHeight;

    // Draw the current video frame (raw)
    rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

    // Download raw frame
    const rawFilename = `raw-frame-${timestamp}.png`;
    rawCanvas.toBlob((blob) => {
      if (blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = rawFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    }, 'image/png');

    // If we have detections, also create annotated version
    if (lastResult?.detections && lastResult.detections.length > 0) {
      // Create a debug canvas to combine video + detections
      const debugCanvas = document.createElement('canvas');
      const debugCtx = debugCanvas.getContext('2d');
      if (!debugCtx) return;

      // Set canvas size to match video
      debugCanvas.width = video.videoWidth;
      debugCanvas.height = video.videoHeight;

      // Draw the current video frame
      debugCtx.drawImage(video, 0, 0, debugCanvas.width, debugCanvas.height);

      // Draw detection overlays
      debugCtx.strokeStyle = '#00ff00';
      debugCtx.lineWidth = 3;
      debugCtx.font = '16px Arial';
      debugCtx.fillStyle = '#00ff00';

      lastResult.detections.forEach((detection: any, index: number) => {
        if (detection.isRotated && detection.corners && detection.corners.length === 4) {
          // Draw OBB corners
          const corners = detection.corners.map((corner: any) => ({
            x: corner.x * debugCanvas.width,
            y: corner.y * debugCanvas.height
          }));

          debugCtx.beginPath();
          debugCtx.moveTo(corners[0].x, corners[0].y);
          for (let i = 1; i < corners.length; i++) {
            debugCtx.lineTo(corners[i].x, corners[i].y);
          }
          debugCtx.closePath();
          debugCtx.stroke();

          // Draw corner numbers
          corners.forEach((corner: any, i: number) => {
            debugCtx.fillText(`${i}`, corner.x + 5, corner.y - 5);
          });

          // Draw confidence
          debugCtx.fillText(
            `${detection.confidence.toFixed(3)}`,
            corners[0].x,
            corners[0].y - 20
          );
        } else {
          // Draw regular bounding box
          const x = detection.boundingBox.x * debugCanvas.width;
          const y = detection.boundingBox.y * debugCanvas.height;
          const w = detection.boundingBox.width * debugCanvas.width;
          const h = detection.boundingBox.height * debugCanvas.height;

          debugCtx.strokeRect(x, y, w, h);
          debugCtx.fillText(`${detection.confidence.toFixed(3)}`, x, y - 5);
        }
      });

      // Download the annotated debug image
      const debugFilename = `debug-frame-${timestamp}.png`;
      
      debugCanvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = debugFilename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }
      }, 'image/png');

      console.log(`📸 Debug frames captured: ${rawFilename} & ${debugFilename}`);
      console.log(`📐 Frame size: ${rawCanvas.width}x${rawCanvas.height}`);
      console.log(`🎯 Detections: ${lastResult.detections.length}`);
    } else {
      console.log(`📸 Raw frame captured: ${rawFilename}`);
      console.log(`📐 Frame size: ${rawCanvas.width}x${rawCanvas.height}`);
      console.log(`🎯 No detections to overlay`);
    }
  };

  // Initialize camera and model on mount
  useEffect(() => {
    console.log('🎬 CameraInterface: Initializing camera and model...');
    console.log('📊 Model state:', modelState);
    initializeCamera();
    loadModel();
    return () => stopCamera();
  }, [initializeCamera, stopCamera, loadModel]);

  // Monitor model state changes
  useEffect(() => {
    console.log('📈 Model state changed:', {
      isLoaded: modelState.isLoaded,
      isLoading: modelState.isLoading,
      error: modelState.error,
      path: modelState.modelPath
    });
    modelLoadedRef.current = modelState.isLoaded;
  }, [modelState]);

  // Render frozen frame to canvas when captured
  useEffect(() => {
    if (frozenFrameData && frozenCanvasRef.current) {
      const canvas = frozenCanvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        canvas.width = frozenFrameData.width;
        canvas.height = frozenFrameData.height;
        ctx.putImageData(frozenFrameData, 0, 0);
      }
    }
  }, [frozenFrameData]);

  // Note: Confidence filtering is now done at render time in DetectionOverlay

  // Attach video element when available
  useEffect(() => {
    if (videoRef.current && cameraState.stream) {
      attachVideoElement(videoRef.current);
    }
  }, [cameraState.stream, attachVideoElement]);

  // Start frame processing when video is ready
  useEffect(() => {
    if (videoRef.current && cameraState.status === 'streaming') {
      const video = videoRef.current;
      
      const handleLoadedMetadata = () => {
        startFrameProcessing();
      };

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      
      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      };
    }
  }, [cameraState.status]);

  const startFrameProcessing = () => {
    const processFrame = () => {
      if (cameraState.status === 'streaming' && modelLoadedRef.current && !shouldSkipFrame() && !isCameraFrozen) {
        captureAndAnalyzeFrame();
      }
      setFrameCount((c) => (c + 1) % 60);
      if (frameCount === 0) {
        console.log('🎞️ Frame loop alive. Model loaded:', modelLoadedRef.current, 'Queue:', isInferenceProcessing, 'Frozen:', isCameraFrozen);
      }
      requestAnimationFrame(processFrame);
    };
    
    processFrame();
  };

  const captureAndAnalyzeFrame = async () => {
    if (!videoRef.current || !canvasRef.current || !modelLoadedRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true } as any) as CanvasRenderingContext2D | null;
    
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data for processing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Run inference if not already processing
    if (!isInferenceProcessing) {
      try {
        await runInference(imageData, {
          skipQueue: true,
          callback: (result) => {
            // Add result to detection system
            detectionSystemRef.current.addDetectionResult(result);
            
            // Analyze positioning
            const positioning = detectionSystemRef.current.analyzeCardPositioning();
            
            // Update detection state based on positioning analysis
            if (positioning.isWellPositioned) {
              setDetectionState('ready');
            } else if (result.detections.length > 0) {
              if (positioning.quality.score > 0.25) {
                // Allow capture when cards are detected with any reasonable quality
                setDetectionState('ready');
              } else if (positioning.quality.score > 0.15) {
                setDetectionState('positioned');
              } else {
                setDetectionState('detected');
              }
            } else {
              setDetectionState('searching');
            }
          }
        });
      } catch (error) {
        console.error('Frame analysis failed:', error);
      }
    }
  };

  const handleCapture = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    // Freeze the camera and capture the current frame
    setIsCameraFrozen(true);
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Capture the current frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setFrozenFrameData(imageData);
    
    // If no detections, still capture the frame but show a message
    if (!lastResult?.detections || lastResult.detections.length === 0) {
      console.log('No cards detected, but capturing frame anyway');
      if (onCapture) {
        onCapture(imageData);
      }
      // Keep camera frozen to show the captured frame
      return;
    }

    setIsProcessing(true);
    
    try {
      
      // Filter detections by confidence threshold (use lower threshold for capture)
      const captureThreshold = Math.min(confidenceThreshold, 0.3); // Use at most 0.3 for capture
      const validDetections = lastResult.detections.filter(
        (detection: any) => detection.confidence >= captureThreshold && 
        CardCropper.isValidCardDetection(detection)
      );

      if (validDetections.length === 0) {
        if (onError) {
          onError('No valid cards detected above confidence threshold');
        }
        return;
      }

      // Start extraction process
      dispatch(startExtraction({
        sourceImageDimensions: {
          width: imageData.width,
          height: imageData.height
        },
        totalDetections: validDetections.length
      }));

      // Extract cards one by one
      const cropResults = CardCropper.extractCards(imageData, validDetections);
      
      for (let i = 0; i < cropResults.length; i++) {
        const cropResult = cropResults[i];
        const detection = validDetections[i];
        
        dispatch(updateExtractionProgress({
          current: i + 1,
          status: `Extracting card ${i + 1} of ${validDetections.length}`
        }));

        const extractedCard = {
          id: `extracted_${Date.now()}_${i}`,
          imageData: cropResult.imageData,
          originalDetection: detection,
          extractedAt: Date.now(),
          dimensions: {
            width: cropResult.croppedWidth,
            height: cropResult.croppedHeight
          }
        };

        dispatch(addExtractedCard(extractedCard));
        
        // Small delay to show progress
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Complete extraction and navigate to extraction view
      dispatch(completeExtraction());
      dispatch(setCurrentView('extraction'));

      // Call original onCapture if provided
      if (onCapture) {
        onCapture(imageData);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Card extraction failed';
      console.error('Card extraction error:', error);
      if (onError) {
        onError(errorMessage);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const resumeCamera = () => {
    setIsCameraFrozen(false);
    setFrozenFrameData(null);
    setIsProcessing(false);
  };

  // Expose resumeCamera function to parent component
  useImperativeHandle(ref, () => ({
    resumeCamera
  }));

  const handleCameraFlip = () => {
    // TODO: Implement camera flip functionality
    console.log('Camera flip requested');
  };

  const handleRetry = () => {
    if (cameraState.error?.name === 'OverconstrainedError') {
      retryWithFallback();
    } else {
      initializeCamera();
    }
  };

  const renderError = () => {
    if (!cameraState.error) return null;

    return (
      <div className="camera-error">
        <div className="error-content">
          <h3>{cameraState.error.name === 'NotAllowedError' ? 'Camera Permission Required' : 'Camera Error'}</h3>
          <p>{cameraState.error.message}</p>
          <button onClick={handleRetry} className="retry-button">
            {cameraState.error.name === 'OverconstrainedError' ? 'Try Lower Quality' : 'Retry'}
          </button>
        </div>
      </div>
    );
  };

  const renderPositioningGuide = () => {
    if (detectionState === 'searching') return null;

    const positioning = detectionSystemRef.current.analyzeCardPositioning();
    
    return (
      <div className={`positioning-guide ${detectionState}`}>
        <p>{positioning.feedback}</p>
      </div>
    );
  };

  if (cameraState.status === 'error') {
    return renderError();
  }

  return (
    <div className="camera-container">
      <div className="camera-viewport">
        <video
          ref={videoRef}
          className="camera-video"
          autoPlay
          playsInline
          muted
          style={{ display: isCameraFrozen ? 'none' : 'block' }}
        />
        
        <canvas
          ref={canvasRef}
          className="processing-canvas"
          style={{ display: 'none' }}
        />
        
        <canvas
          ref={frozenCanvasRef}
          className="frozen-frame-canvas"
          style={{ 
            display: isCameraFrozen ? 'block' : 'none',
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }}
        />
        
        <DetectionOverlay
          detections={lastResult?.detections || []}
          quality={lastResult ? detectionSystemRef.current.analyzeCardPositioning().quality : undefined}
          videoElement={videoRef.current}
          className="detection-overlay"
          confidenceThreshold={confidenceThreshold}
        />
        
        {isCameraFrozen && isProcessing && (
          <div className="processing-overlay">
            <div className="processing-content">
              <div className="loading-spinner large" />
              <h3>Processing Frame</h3>
              <p>Extracting cards from captured image...</p>
            </div>
          </div>
        )}
        
        {isCameraFrozen && !isProcessing && (
          <div className="frame-captured-overlay">
            <div className="frame-captured-content">
              <h3>Frame Captured</h3>
              <p>Ready for processing</p>
              <button onClick={resumeCamera} className="resume-camera-btn">
                Resume Camera
              </button>
            </div>
          </div>
        )}
      </div>

      {/* lightweight debug status */}
      <div className="debug-panel">
        <div>Model: {modelState.isLoaded ? 'loaded' : (modelState.isLoading ? 'loading' : 'idle')}</div>
        <div>NMS: {modelState.isLoaded ? 'active' : 'inactive'}</div>
        <div>Detections: {lastResult?.detections?.filter((d: any) => d.confidence >= confidenceThreshold).length ?? 0} / {lastResult?.detections?.length ?? 0}</div>
        <div>State: {detectionState} | Q: {detectionSystemRef.current.analyzeCardPositioning().quality.score.toFixed(2)}</div>
        <button 
          onClick={captureDebugFrame}
          disabled={cameraState.status !== 'streaming' && cameraState.status !== 'processing'}
        >
          📸 Debug
        </button>
      </div>

      {/* Confidence threshold slider */}
      <div className="confidence-slider-container">
        <div className="confidence-slider-title">Confidence Filter</div>
        <div className="confidence-slider-controls">
          <span className="confidence-slider-label">0.0</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="confidence-slider"
          />
          <span className="confidence-slider-label">1.0</span>
        </div>
        <div className="confidence-value-display">
          {(confidenceThreshold * 100).toFixed(0)}%
        </div>
      </div>

      <div className="camera-controls">
        <button
          className={`capture-button ${detectionState} ${isProcessing ? 'processing' : ''}`}
          onClick={handleCapture}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <div className="loading-spinner" />
          ) : (
            <div className="camera-icon" />
          )}
        </button>

        <button
          className="secondary-control flip-camera"
          onClick={handleCameraFlip}
          disabled={cameraState.status !== 'streaming'}
        >
          <div className="flip-icon" />
        </button>
      </div>

      {renderPositioningGuide()}

      {cameraState.status === 'initializing' && (
        <div className="camera-loading">
          <div className="loading-spinner" />
          <p>Initializing camera...</p>
        </div>
      )}

      {!modelState.isLoaded && !modelState.isLoading && (
        <div className="model-status">
          <p>Model not loaded - Detection disabled</p>
        </div>
      )}

      {modelState.isLoading && (
        <div className="model-loading">
          <div className="loading-spinner" />
          <p>Loading ML model...</p>
        </div>
      )}
    </div>
  );
});
