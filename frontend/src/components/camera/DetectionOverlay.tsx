import React, { useRef, useEffect } from 'react';
import { CardDetection } from '../../store/slices/inferenceSlice';
import { DetectionQuality } from '../../utils/CardDetectionSystem';

interface DetectionOverlayProps {
  detections: CardDetection[];
  quality?: DetectionQuality;
  videoElement: HTMLVideoElement | null;
  className?: string;
  confidenceThreshold?: number;
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  detections,
  quality,
  videoElement,
  className = 'detection-overlay',
  confidenceThreshold = 0.0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoElement) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const updateCanvasSize = () => {
      // Get the actual rendered size of the video element (after CSS scaling)
      const videoRect = videoElement.getBoundingClientRect();
      const canvasWidth = videoRect.width;
      const canvasHeight = videoRect.height;

      // Calculate the visible video area when using object-fit: cover
      const videoNativeWidth = videoElement.videoWidth;
      const videoNativeHeight = videoElement.videoHeight;
      const videoNativeAspect = videoNativeWidth / videoNativeHeight;
      const containerAspect = canvasWidth / canvasHeight;

      let visibleVideoInfo = {
        scaleX: canvasWidth / videoNativeWidth,
        scaleY: canvasHeight / videoNativeHeight,
        offsetX: 0,
        offsetY: 0,
        visibleWidth: videoNativeWidth,
        visibleHeight: videoNativeHeight
      };

      // With object-fit: cover, the video is scaled to fill the container
      // and cropped if aspect ratios don't match
      if (videoNativeAspect > containerAspect) {
        // Video is wider than container - crop horizontally
        const scaleFactor = canvasHeight / videoNativeHeight;
        const scaledVideoWidth = videoNativeWidth * scaleFactor;
        const cropAmount = (scaledVideoWidth - canvasWidth) / 2;
        
        visibleVideoInfo = {
          scaleX: scaleFactor,
          scaleY: scaleFactor,
          offsetX: cropAmount / scaleFactor, // Offset in native video coordinates
          offsetY: 0,
          visibleWidth: canvasWidth / scaleFactor, // Visible width in native coordinates
          visibleHeight: videoNativeHeight
        };
      } else if (videoNativeAspect < containerAspect) {
        // Video is taller than container - crop vertically
        const scaleFactor = canvasWidth / videoNativeWidth;
        const scaledVideoHeight = videoNativeHeight * scaleFactor;
        const cropAmount = (scaledVideoHeight - canvasHeight) / 2;
        
        visibleVideoInfo = {
          scaleX: scaleFactor,
          scaleY: scaleFactor,
          offsetX: 0,
          offsetY: cropAmount / scaleFactor, // Offset in native video coordinates
          visibleWidth: videoNativeWidth,
          visibleHeight: canvasHeight / scaleFactor // Visible height in native coordinates
        };
      }

      // Debug logging for scaling issues
      console.log(`🔍 DetectionOverlay updateCanvasSize:`, {
        videoNative: `${videoNativeWidth}x${videoNativeHeight}`,
        videoRendered: `${canvasWidth.toFixed(1)}x${canvasHeight.toFixed(1)}`,
        canvasBefore: `${canvas.width}x${canvas.height}`,
        windowSize: `${window.innerWidth}x${window.innerHeight}`,
        aspectRatios: `native:${videoNativeAspect.toFixed(2)} container:${containerAspect.toFixed(2)}`,
        visibleArea: `${visibleVideoInfo.visibleWidth.toFixed(1)}x${visibleVideoInfo.visibleHeight.toFixed(1)}`,
        offset: `${visibleVideoInfo.offsetX.toFixed(1)},${visibleVideoInfo.offsetY.toFixed(1)}`,
        scale: `${visibleVideoInfo.scaleX.toFixed(3)}`
      });

      // Store visible video info for use in drawing functions
      (canvas as any).visibleVideoInfo = visibleVideoInfo;

      // Set canvas size to match the rendered video size
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      // Clear canvas
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // Filter detections by confidence threshold
      const filteredDetections = detections.filter(detection => detection.confidence >= confidenceThreshold);

      // Debug canvas and detection info
      if (filteredDetections.length > 0) {
        console.log(`🖼️ DetectionOverlay: Canvas ${canvasWidth}x${canvasHeight}, ${detections.length} total detections, ${filteredDetections.length} after confidence filter (>=${confidenceThreshold})`);
        console.log(`🖼️ Video native size: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
        console.log(`🖼️ Video rendered size: ${canvasWidth}x${canvasHeight}`);
        
        filteredDetections.slice(0, 1).forEach((detection, index) => { // Only show first detection for clarity
          console.log(`  Detection ${index}:`, {
            boundingBox: `x:${detection.boundingBox.x.toFixed(1)}, y:${detection.boundingBox.y.toFixed(1)}, w:${detection.boundingBox.width.toFixed(1)}, h:${detection.boundingBox.height.toFixed(1)}`,
            corners: detection.corners ? detection.corners.map((c, i) => `C${i}:(${c.x.toFixed(1)},${c.y.toFixed(1)})`).join(' ') : 'none',
            confidence: detection.confidence.toFixed(3),
            isRotated: detection.isRotated
          });
        });
      }

      // Draw filtered detections
      filteredDetections.forEach((detection, index) => {
        drawDetection(ctx, detection, index === 0, quality, (canvas as any).visibleVideoInfo);
      });

      // Draw positioning guides if we have filtered detections
      if (filteredDetections.length > 0) {
        drawPositioningGuides(ctx, canvasWidth, canvasHeight, quality);
      }
    };

    // Initial render
    updateCanvasSize();

    // Add resize listener to handle window resizing
    const handleResize = () => {
      // Add a small delay to ensure video element has updated its size
      setTimeout(() => {
        updateCanvasSize();
      }, 50);
    };

    window.addEventListener('resize', handleResize);
    
    // Also use ResizeObserver for more reliable detection of video element size changes
    let resizeObserver: ResizeObserver | null = null;
    if (window.ResizeObserver) {
      resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.target === videoElement) {
            console.log('🔍 Video element resized via ResizeObserver');
            setTimeout(() => {
              updateCanvasSize();
            }, 10);
            break;
          }
        }
      });
      resizeObserver.observe(videoElement);
    }
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [detections, quality, videoElement, confidenceThreshold]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 2,
      }}
    />
  );
};

function drawDetection(
  ctx: CanvasRenderingContext2D,
  detection: CardDetection,
  isPrimary: boolean,
  quality: DetectionQuality | undefined,
  visibleVideoInfo: any
): void {
  const { boundingBox, confidence, corners, isRotated } = detection;

  // Determine colors based on quality and confidence
  let strokeColor = '#ff0000'; // Red for poor quality
  let fillColor = 'rgba(255, 0, 0, 0.1)';

  if (quality) {
    if (quality.score > 0.8) {
      strokeColor = '#00ff00'; // Green for good quality
      fillColor = 'rgba(0, 255, 0, 0.1)';
    } else if (quality.score > 0.6) {
      strokeColor = '#ffff00'; // Yellow for medium quality
      fillColor = 'rgba(255, 255, 0, 0.1)';
    } else if (quality.score > 0.4) {
      strokeColor = '#ff8800'; // Orange for low quality
      fillColor = 'rgba(255, 136, 0, 0.1)';
    }
  } else if (confidence > 0.8) {
    strokeColor = '#00ff00';
    fillColor = 'rgba(0, 255, 0, 0.1)';
  } else if (confidence > 0.6) {
    strokeColor = '#ffff00';
    fillColor = 'rgba(255, 255, 0, 0.1)';
  }

  // Set drawing properties
  ctx.strokeStyle = strokeColor;
  ctx.fillStyle = fillColor;
  ctx.lineWidth = isPrimary ? 3 : 2;

  // Add glow effect for primary detection
  if (isPrimary) {
    ctx.shadowColor = strokeColor;
    ctx.shadowBlur = 10;
  }

  // Check if detection is within the visible video area
  const isDetectionVisible = (x: number, y: number, width: number, height: number) => {
    return x + width > visibleVideoInfo.offsetX && 
           x < visibleVideoInfo.offsetX + visibleVideoInfo.visibleWidth &&
           y + height > visibleVideoInfo.offsetY && 
           y < visibleVideoInfo.offsetY + visibleVideoInfo.visibleHeight;
  };

  // Transform coordinates from native video space to visible canvas space
  const transformCoordinate = (x: number, y: number) => {
    // Adjust for visible area offset
    const adjustedX = x - visibleVideoInfo.offsetX;
    const adjustedY = y - visibleVideoInfo.offsetY;
    
    // Scale to canvas coordinates
    const canvasX = adjustedX * visibleVideoInfo.scaleX;
    const canvasY = adjustedY * visibleVideoInfo.scaleY;
    
    return { x: canvasX, y: canvasY };
  };

  if (isRotated && corners && corners.length === 4) {
    // Check if any corner is visible
    const visibleCorners = corners.filter(corner => 
      isDetectionVisible(corner.x, corner.y, 0, 0)
    );
    
    if (visibleCorners.length === 0) return; // Skip if completely outside visible area
    
    // Transform corners to canvas coordinates
    const canvasCorners = corners.map(corner => transformCoordinate(corner.x, corner.y));
    
    // Draw rotated bounding box using corner points
    drawRotatedBoundingBox(ctx, canvasCorners, fillColor, strokeColor);
    
    // Draw corner markers for rotated box
    drawRotatedCorners(ctx, canvasCorners, strokeColor, isPrimary);
    
    // Draw confidence label at first corner
    if (isPrimary) {
      drawConfidenceLabel(ctx, canvasCorners[0].x, canvasCorners[0].y, confidence, strokeColor);
    }
  } else {
    // Check if bounding box is visible
    if (!isDetectionVisible(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)) {
      return; // Skip if completely outside visible area
    }
    
    // Transform bounding box coordinates
    const topLeft = transformCoordinate(boundingBox.x, boundingBox.y);
    const bottomRight = transformCoordinate(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height);
    
    const canvasX = Math.min(topLeft.x, bottomRight.x);
    const canvasY = Math.min(topLeft.y, bottomRight.y);
    const canvasBoxWidth = Math.abs(bottomRight.x - topLeft.x);
    const canvasBoxHeight = Math.abs(bottomRight.y - topLeft.y);
    
    ctx.fillRect(canvasX, canvasY, canvasBoxWidth, canvasBoxHeight);
    ctx.strokeRect(canvasX, canvasY, canvasBoxWidth, canvasBoxHeight);
    
    // Draw corners for better visibility
    drawCorners(ctx, canvasX, canvasY, canvasBoxWidth, canvasBoxHeight, strokeColor, isPrimary);
    
    // Draw confidence label
    if (isPrimary) {
      drawConfidenceLabel(ctx, canvasX, canvasY, confidence, strokeColor);
    }
  }

  // Reset shadow
  ctx.shadowBlur = 0;
}

function drawCorners(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  color: string,
  isPrimary: boolean
): void {
  const cornerLength = isPrimary ? 20 : 15;
  const cornerWidth = isPrimary ? 4 : 3;

  ctx.strokeStyle = color;
  ctx.lineWidth = cornerWidth;
  ctx.lineCap = 'round';

  // Top-left corner
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLength);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLength, y);
  ctx.stroke();

  // Top-right corner
  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + cornerLength);
  ctx.stroke();

  // Bottom-right corner
  ctx.beginPath();
  ctx.moveTo(x + width, y + height - cornerLength);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x + width - cornerLength, y + height);
  ctx.stroke();

  // Bottom-left corner
  ctx.beginPath();
  ctx.moveTo(x + cornerLength, y + height);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x, y + height - cornerLength);
  ctx.stroke();
}

function drawConfidenceLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  confidence: number,
  color: string
): void {
  const label = `${Math.round(confidence * 100)}%`;
  const padding = 8;
  const fontSize = 14;

  ctx.font = `${fontSize}px Arial`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';

  const textMetrics = ctx.measureText(label);
  const labelWidth = textMetrics.width + padding * 2;
  const labelHeight = fontSize + padding * 2;

  // Position label above the bounding box
  const labelX = x;
  const labelY = Math.max(0, y - labelHeight - 5);

  // Draw label background
  ctx.fillStyle = color;
  ctx.fillRect(labelX, labelY, labelWidth, labelHeight);

  // Draw label text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, labelX + padding, labelY + padding);
}

function drawPositioningGuides(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  quality?: DetectionQuality
): void {
  // Draw center guides
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;
  const guideLength = 40;

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);

  // Horizontal center line
  ctx.beginPath();
  ctx.moveTo(centerX - guideLength, centerY);
  ctx.lineTo(centerX + guideLength, centerY);
  ctx.stroke();

  // Vertical center line
  ctx.beginPath();
  ctx.moveTo(centerX, centerY - guideLength);
  ctx.lineTo(centerX, centerY + guideLength);
  ctx.stroke();

  // Reset line dash
  ctx.setLineDash([]);

  // Draw ideal positioning area
  const idealWidth = canvasWidth * 0.6;
  const idealHeight = canvasHeight * 0.4;
  const idealX = (canvasWidth - idealWidth) / 2;
  const idealY = (canvasHeight - idealHeight) / 2;

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([10, 10]);
  ctx.strokeRect(idealX, idealY, idealWidth, idealHeight);
  ctx.setLineDash([]);

  // Draw quality indicator
  if (quality) {
    drawQualityIndicator(ctx, canvasWidth, canvasHeight, quality);
  }
}

function drawQualityIndicator(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  quality: DetectionQuality
): void {
  const indicatorSize = 60;
  const x = canvasWidth - indicatorSize - 20;
  const y = 20;

  // Draw background circle
  ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
  ctx.beginPath();
  ctx.arc(x + indicatorSize / 2, y + indicatorSize / 2, indicatorSize / 2, 0, Math.PI * 2);
  ctx.fill();

  // Draw quality arc
  const centerX = x + indicatorSize / 2;
  const centerY = y + indicatorSize / 2;
  const radius = indicatorSize / 2 - 5;
  const startAngle = -Math.PI / 2;
  const endAngle = startAngle + (Math.PI * 2 * quality.score);

  let arcColor = '#ff0000';
  if (quality.score > 0.8) arcColor = '#00ff00';
  else if (quality.score > 0.6) arcColor = '#ffff00';
  else if (quality.score > 0.4) arcColor = '#ff8800';

  ctx.strokeStyle = arcColor;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, startAngle, endAngle);
  ctx.stroke();

  // Draw quality percentage
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${Math.round(quality.score * 100)}%`, centerX, centerY);
}

function drawRotatedBoundingBox(
  ctx: CanvasRenderingContext2D,
  corners: Array<{ x: number; y: number }>,
  fillColor: string,
  strokeColor: string
): void {
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  
  for (let i = 1; i < corners.length; i++) {
    ctx.lineTo(corners[i].x, corners[i].y);
  }
  
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawRotatedCorners(
  ctx: CanvasRenderingContext2D,
  corners: Array<{ x: number; y: number }>,
  color: string,
  isPrimary: boolean
): void {
  const cornerRadius = isPrimary ? 6 : 4;
  const cornerWidth = isPrimary ? 3 : 2;

  ctx.fillStyle = color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = cornerWidth;

  corners.forEach((corner, index) => {
    // Draw corner circle
    ctx.beginPath();
    ctx.arc(corner.x, corner.y, cornerRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    
    // Draw corner number for debugging (optional)
    if (isPrimary) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText((index + 1).toString(), corner.x, corner.y);
      ctx.fillStyle = color; // Reset fill color
    }
  });
}
