import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import { useCamera } from '../../hooks/useCamera';
import { useInference } from '../../hooks/useInference';
import { useAppDispatch } from '../../hooks/redux';
import { DetectionOverlay } from './DetectionOverlay';
import { CardDetectionSystem } from '../../utils/CardDetectionSystem';
import { CardCropper } from '../../utils/CardCropper';
import { EnhancedCardCropper } from '../../utils/EnhancedCardCropper';
import { MultiFrameCapture, MultiFrameCaptureResult } from '../../utils/MultiFrameCapture';
import { DetectionFusion, FusedDetection } from '../../utils/DetectionFusion';
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
  const [processingStep, setProcessingStep] = useState<string>('');
  const [extractionProgress, setExtractionProgress] = useState<{current: number, total: number}>({current: 0, total: 0});
  const [multiFrameCapture, setMultiFrameCapture] = useState<MultiFrameCaptureResult | null>(null);
  const [fusedDetections, setFusedDetections] = useState<FusedDetection[]>([]);
  const [isCapturingFrames, setIsCapturingFrames] = useState(false);
  const modelLoadedRef = useRef<boolean>(false);
  const isCapturingFramesRef = useRef<boolean>(false);
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
    runInference,
    runBatchInference
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
        
        // Draw the clean captured frame
        ctx.putImageData(frozenFrameData, 0, 0);
        
        // Overlay detection boxes if we have detections
        if (lastResult?.detections && lastResult.detections.length > 0) {
          // Filter detections by confidence threshold
          const captureThreshold = Math.min(confidenceThreshold, 0.3);
          const validDetections = lastResult.detections.filter(
            (detection: any) => detection.confidence >= captureThreshold
          );
          
          // Draw detection overlays
          ctx.strokeStyle = '#00ff00';
          ctx.lineWidth = 3;
          ctx.font = '16px Arial';
          ctx.fillStyle = '#00ff00';
          ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
          ctx.shadowBlur = 4;

          validDetections.forEach((detection: any, index: number) => {
            if (detection.isRotated && detection.corners && detection.corners.length === 4) {
              // Draw OBB corners
              const corners = detection.corners.map((corner: any) => ({
                x: corner.x * canvas.width,
                y: corner.y * canvas.height
              }));

              ctx.beginPath();
              ctx.moveTo(corners[0].x, corners[0].y);
              for (let i = 1; i < corners.length; i++) {
                ctx.lineTo(corners[i].x, corners[i].y);
              }
              ctx.closePath();
              ctx.stroke();

              // Draw confidence
              ctx.fillText(
                `Card ${index + 1}: ${(detection.confidence * 100).toFixed(1)}%`,
                corners[0].x,
                corners[0].y - 10
              );
            } else {
              // Draw regular bounding box
              const x = detection.boundingBox.x * canvas.width;
              const y = detection.boundingBox.y * canvas.height;
              const w = detection.boundingBox.width * canvas.width;
              const h = detection.boundingBox.height * canvas.height;

              ctx.strokeRect(x, y, w, h);
              ctx.fillText(
                `Card ${index + 1}: ${(detection.confidence * 100).toFixed(1)}%`,
                x,
                y - 10
              );
            }
          });
          
          // Reset shadow
          ctx.shadowBlur = 0;
        }
      }
    }
  }, [frozenFrameData, lastResult, confidenceThreshold]);

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
      // Use ref for immediate state checking to avoid React state update delays
      const shouldProcess = cameraState.status === 'streaming' && modelLoadedRef.current && !shouldSkipFrame() && !isCameraFrozen && !isCapturingFramesRef.current;
      
      if (shouldProcess) {
        captureAndAnalyzeFrame();
      }
      
      setFrameCount((c) => (c + 1) % 60);
      if (frameCount === 0) {
        console.log('🎞️ Frame loop alive. Model loaded:', modelLoadedRef.current, 'Queue:', isInferenceProcessing, 'Frozen:', isCameraFrozen, 'Capturing:', isCapturingFramesRef.current);
      }
      
      // Log when live detection is blocked during capture
      if (!shouldProcess && isCapturingFramesRef.current && frameCount % 30 === 0) {
        console.log('🚫 Live detection BLOCKED - capture in progress');
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
    
    // Immediately stop live detection using ref for synchronous access
    isCapturingFramesRef.current = true;
    
    // Freeze the camera and start instant multi-frame capture
    setIsCameraFrozen(true);
    setIsCapturingFrames(true); // Also set state for UI consistency
    setProcessingStep('Live detection stopped - capturing frames...');
    setIsProcessing(true);
    
    console.log('🛑 IMMEDIATELY stopped live detection for multi-frame capture');
    console.log('👻 Detection overlay will be hidden during capture');
    
    // Much shorter delay since we're using ref for immediate state change
    await new Promise(resolve => setTimeout(resolve, 16)); // Just one frame
    
    setProcessingStep('Instantly capturing 3 frames...');
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    try {
      // Phase 1: Capture all frames instantly with NO delays
      console.log('🚀 Starting INSTANT multi-frame capture...');
      const captureResult = await MultiFrameCapture.captureFramesInstantWithValidation(video, {
        frameCount: 3,
        canvas: canvas,
        maxAttempts: 2,
        stabilityThreshold: 0.6 // More lenient since frames are captured instantly
      });
      
      setMultiFrameCapture(captureResult);
      
      // Use middle frame for display immediately
      const middleFrame = MultiFrameCapture.getMiddleFrame(captureResult);
      setFrozenFrameData(middleFrame.imageData);
      
      console.log(`✅ All ${captureResult.frames.length} frames captured INSTANTLY in ${captureResult.captureDuration.toFixed(2)}ms`);
      
      // Phase 2: Run inference on all captured frames (now that capture is complete)
      setProcessingStep('Running inference on captured frames...');
      console.log('🔍 Running batch inference on all captured frames...');
      
      const frameImageData = MultiFrameCapture.getFrameImageData(captureResult);
      const inferenceResults = await runBatchInference(frameImageData, (completed: number, total: number) => {
        setProcessingStep(`Analyzing frame ${completed}/${total}...`);
      });
      
      if (inferenceResults.length === 0) {
        console.log('No inference results, but capturing frame anyway');
        setProcessingStep('No cards detected');
        if (onCapture) {
          onCapture(middleFrame.imageData);
        }
        setIsProcessing(false);
        return;
      }
      
      setProcessingStep('Fusing detection results...');
      
      // Fuse detection results from all frames with enhanced parameters
      console.log('🔄 Fusing detection results with enhanced multi-frame analysis...');
      const fusionResult = DetectionFusion.fuseDetections(inferenceResults, {
        iouThreshold: 0.5,
        minFrameCount: 1, // Allow detections from single frames
        confidenceWeight: 0.7,
        positionWeight: 0.3,
        enableTemporalConsistency: true,
        preserveSingleFrameDetections: true,
        multiFrameBoostFactor: 0.15 // Increased confidence boost for multi-frame detections
      });
      
      // Filter for high-quality detections with enhanced criteria
      const highQualityDetections = DetectionFusion.filterHighQualityDetections(
        fusionResult.fusedDetections,
        {
          minFrameCount: 1, // Allow single-frame detections
          minAverageConfidence: 0.25,
          maxConfidenceVariance: 0.4,
          minTemporalConsistency: 0.3, // Require reasonable consistency for multi-frame
          prioritizeMultiFrame: true // Sort multi-frame detections first
        }
      );
      
      setFusedDetections(highQualityDetections);
      
      // Enhanced fusion analysis and logging
      const fusionAnalysis = DetectionFusion.analyzeFusionQuality(fusionResult);
      console.log(`✅ Enhanced fusion complete: ${fusionResult.fusedDetections.length} total, ${highQualityDetections.length} high-quality`);
      console.log(`📊 ${fusionAnalysis.summary}`);
      if (fusionAnalysis.recommendations.length > 0) {
        console.log(`💡 Recommendations:`, fusionAnalysis.recommendations);
      }
      
      // Convert fused detections back to regular format for extraction
      const finalDetections = DetectionFusion.convertToCardDetections(highQualityDetections);
      
      // Filter detections by confidence threshold (use lower threshold for capture)
      const captureThreshold = Math.min(confidenceThreshold, 0.3);
      const validDetections = finalDetections.filter(
        (detection: any) => detection.confidence >= captureThreshold && 
        EnhancedCardCropper.isValidCardDetection(detection, captureThreshold)
      );

      if (validDetections.length === 0) {
        setProcessingStep('No valid cards detected');
        if (onError) {
          onError('No valid cards detected above confidence threshold');
        }
        setIsProcessing(false);
        return;
      }

      // Set up progress tracking
      setExtractionProgress({current: 0, total: validDetections.length});
      setProcessingStep(`Found ${validDetections.length} card${validDetections.length === 1 ? '' : 's'} (from ${fusionResult.fusedDetections.length} fused)`);
      
      // Small delay to show the detection count
      await new Promise(resolve => setTimeout(resolve, 500));

      // Start extraction process using middle frame
      dispatch(startExtraction({
        sourceImageDimensions: {
          width: middleFrame.imageData.width,
          height: middleFrame.imageData.height
        },
        totalDetections: validDetections.length
      }));

      setProcessingStep('Preparing extraction...');
      await new Promise(resolve => setTimeout(resolve, 300));

      // Extract cards using high-resolution extraction with video element
      console.log('🎯 Starting high-resolution card extraction...');
      console.log(`📐 Middle frame dimensions: ${middleFrame.imageData.width}x${middleFrame.imageData.height}`);
      console.log(`🎥 Video ref available: ${!!videoRef.current}`);
      
      if (videoRef.current) {
        console.log(`📹 Video element info:`);
        console.log(`   - videoWidth: ${videoRef.current.videoWidth}`);
        console.log(`   - videoHeight: ${videoRef.current.videoHeight}`);
        console.log(`   - readyState: ${videoRef.current.readyState}`);
        console.log(`   - currentTime: ${videoRef.current.currentTime}`);
      }
      
      const cropResults = await EnhancedCardCropper.extractFromCameraFrame(
        middleFrame.imageData,
        validDetections,
        videoRef.current || undefined,
        {
          modelInputSize: { width: middleFrame.imageData.width, height: middleFrame.imageData.height },
          paddingRatio: 0.1, // 10% padding for better extraction (minimum 20px will be applied)
          enablePerspectiveCorrection: true
        }
      );
      
      console.log(`🎯 Extraction complete: ${cropResults.length} cards extracted`);
      cropResults.forEach((result, index) => {
        console.log(`   Card ${index + 1}: ${result.extractedWidth}x${result.extractedHeight} (method: ${result.metadata.extractionMethod}, high-res: ${result.metadata.isHighResolution})`);
      });
      
      for (let i = 0; i < cropResults.length; i++) {
        const cropResult = cropResults[i];
        const detection = validDetections[i];
        
        // Update local progress state
        setExtractionProgress({current: i + 1, total: validDetections.length});
        setProcessingStep(`Extracting card ${i + 1} of ${validDetections.length}...`);
        
        dispatch(updateExtractionProgress({
          current: i + 1,
          status: `Extracting card ${i + 1} of ${validDetections.length}`
        }));

        // Get extraction quality metrics
        const qualityMetrics = EnhancedCardCropper.getExtractionQuality(cropResult);
        
        const extractedCard = {
          id: `extracted_${Date.now()}_${i}`,
          imageData: cropResult.imageData,
          originalDetection: detection,
          extractedAt: Date.now(),
          dimensions: {
            width: cropResult.croppedWidth,
            height: cropResult.croppedHeight
          },
          // Enhanced metadata from high-resolution extraction
          extractionMetadata: cropResult.metadata,
          qualityScore: qualityMetrics.score,
          qualityFactors: qualityMetrics.factors
        };

        dispatch(addExtractedCard(extractedCard));
        
        // Longer delay to show progress more clearly
        await new Promise(resolve => setTimeout(resolve, 400));
      }

      setProcessingStep('Finalizing extraction...');
      await new Promise(resolve => setTimeout(resolve, 500));

      // Complete extraction and navigate to extraction view
      dispatch(completeExtraction());
      dispatch(setCurrentView('extraction'));

      // Call original onCapture if provided
      if (onCapture) {
        onCapture(middleFrame.imageData);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Multi-frame capture failed';
      console.error('Multi-frame capture error:', error);
      if (onError) {
        onError(errorMessage);
      }
    } finally {
      setIsProcessing(false);
      setIsCapturingFrames(false); // Resume live detection
      isCapturingFramesRef.current = false; // Also reset ref
      console.log('▶️ Resuming live detection after capture...');
      console.log('👁️ Detection overlay will be shown again');
    }
  };

  const resumeCamera = () => {
    setIsCameraFrozen(false);
    setFrozenFrameData(null);
    setIsProcessing(false);
    setProcessingStep('');
    setExtractionProgress({current: 0, total: 0});
    setMultiFrameCapture(null);
    setFusedDetections([]);
    setIsCapturingFrames(false); // Ensure live detection resumes
    isCapturingFramesRef.current = false; // Also reset ref
    console.log('▶️ Resuming live detection after manual camera resume...');
    console.log('👁️ Detection overlay will be shown again after manual resume');
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
        
        {!isCapturingFrames && (
          <DetectionOverlay
            detections={lastResult?.detections || []}
            quality={lastResult ? detectionSystemRef.current.analyzeCardPositioning().quality : undefined}
            videoElement={videoRef.current}
            className="detection-overlay"
            confidenceThreshold={confidenceThreshold}
          />
        )}
        
        {isCameraFrozen && isProcessing && (
          <div className="processing-overlay">
            <div className="processing-content">
              <div className="loading-spinner large" />
              <h3>Processing Frame</h3>
              <p className="processing-step">{processingStep}</p>
              {extractionProgress.total > 0 && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${(extractionProgress.current / extractionProgress.total) * 100}%` 
                      }}
                    />
                  </div>
                  <p className="progress-text">
                    {extractionProgress.current} of {extractionProgress.total} cards processed
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
        
        {isCameraFrozen && !isProcessing && (
          <div className="frame-captured-overlay">
            <div className="frame-captured-content">
              <h3>Instant Multi-Frame Capture Complete</h3>
              <p className="capture-status">{processingStep || 'Ready for processing'}</p>
              {multiFrameCapture && (
                <p className="capture-info">
                  {multiFrameCapture.frames.length} frames captured in {multiFrameCapture.captureDuration.toFixed(0)}ms
                  {multiFrameCapture.frames.length > 1 && (
                    <span> (avg: {(multiFrameCapture.captureDuration / (multiFrameCapture.frames.length - 1)).toFixed(0)}ms between frames)</span>
                  )}
                </p>
              )}
              {fusedDetections.length > 0 && (
                <p className="detection-count">
                  {fusedDetections.length} cards detected (fused from multiple frames)
                </p>
              )}
              {lastResult?.detections && lastResult.detections.length > 0 && fusedDetections.length === 0 && (
                <p className="detection-count">
                  {lastResult.detections.filter((d: any) => d.confidence >= Math.min(confidenceThreshold, 0.3)).length} cards detected
                </p>
              )}
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
        {fusedDetections.length > 0 && (
          <div>Fused: {fusedDetections.length} cards (multi: {fusedDetections.filter(d => d.frameCount > 1).length}, avg frames: {(fusedDetections.reduce((sum, d) => sum + d.frameCount, 0) / fusedDetections.length).toFixed(1)}, consistency: {(fusedDetections.reduce((sum, d) => sum + d.temporalConsistency, 0) / fusedDetections.length * 100).toFixed(0)}%)</div>
        )}
        {multiFrameCapture && (
          <div>Frames: {multiFrameCapture.frames.length} captured in {multiFrameCapture.captureDuration.toFixed(0)}ms</div>
        )}
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

CameraInterface.displayName = 'CameraInterface';
