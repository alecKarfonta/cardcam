import React, { useRef, useEffect } from 'react';
import { CardDetection } from '../../store/slices/inferenceSlice';
import { DetectionQuality } from '../../utils/CardDetectionSystem';

interface DetectionOverlayProps {
  detections: CardDetection[];
  quality?: DetectionQuality;
  canvasWidth: number;
  canvasHeight: number;
  className?: string;
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  detections,
  quality,
  canvasWidth,
  canvasHeight,
  className = 'detection-overlay',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Debug canvas and detection info
    if (detections.length > 0) {
      console.log(`🖼️ DetectionOverlay: Canvas ${canvasWidth}x${canvasHeight}, ${detections.length} detections`);
      console.log(`🖼️ Actual canvas element size: ${canvas.width}x${canvas.height}`);
      console.log(`🖼️ Scale factors: scaleX=${(canvas.width / canvasWidth).toFixed(3)}, scaleY=${(canvas.height / canvasHeight).toFixed(3)}`);
      
      detections.slice(0, 1).forEach((detection, index) => { // Only show first detection for clarity
        console.log(`  Detection ${index}:`, {
          boundingBox: `x:${detection.boundingBox.x.toFixed(1)}, y:${detection.boundingBox.y.toFixed(1)}, w:${detection.boundingBox.width.toFixed(1)}, h:${detection.boundingBox.height.toFixed(1)}`,
          corners: detection.corners ? detection.corners.map((c, i) => `C${i}:(${c.x.toFixed(1)},${c.y.toFixed(1)})`).join(' ') : 'none',
          confidence: detection.confidence.toFixed(3),
          isRotated: detection.isRotated
        });
        
        // Show what the coordinates will be after scaling to canvas
        if (detection.corners && detection.corners.length >= 4) {
          const scaleX = canvas.width / canvasWidth;
          const scaleY = canvas.height / canvasHeight;
          const scaledCorners = detection.corners.map((corner, i) => ({
            i,
            x: corner.x * scaleX,
            y: corner.y * scaleY
          }));
          console.log(`    Final canvas corners: ${scaledCorners.map(c => `C${c.i}:(${c.x.toFixed(1)},${c.y.toFixed(1)})`).join(' ')}`);
          console.log(`    Canvas bounds: width=${canvas.width}, height=${canvas.height}`);
        }
      });
    }

    // Draw detections
    detections.forEach((detection, index) => {
      drawDetection(ctx, detection, index === 0, quality, canvasWidth, canvasHeight);
    });

    // Draw positioning guides if we have detections
    if (detections.length > 0) {
      drawPositioningGuides(ctx, canvasWidth, canvasHeight, quality);
    }
  }, [detections, quality, canvasWidth, canvasHeight]);

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
  videoWidth: number,
  videoHeight: number
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

  if (isRotated && corners && corners.length === 4) {
    // Corners are already in image pixel coordinates from convertPredictionToDetections
    // We need to scale them to canvas coordinates
    // videoWidth/videoHeight are the video dimensions, ctx.canvas.width/height are the actual canvas size
    const scaleX = ctx.canvas.width / videoWidth;
    const scaleY = ctx.canvas.height / videoHeight;
    
    const canvasCorners = corners.map(corner => ({
      x: corner.x * scaleX,
      y: corner.y * scaleY
    }));
    
    // FIX: Camera is horizontally mirrored - flip X coordinates
    canvasCorners.forEach(corner => { corner.x = ctx.canvas.width - corner.x; });
    
    // Draw rotated bounding box using corner points
    drawRotatedBoundingBox(ctx, canvasCorners, fillColor, strokeColor);
    
    // Draw corner markers for rotated box
    drawRotatedCorners(ctx, canvasCorners, strokeColor, isPrimary);
    
    // Draw confidence label at first corner
    if (isPrimary) {
      drawConfidenceLabel(ctx, canvasCorners[0].x, canvasCorners[0].y, confidence, strokeColor);
    }
  } else {
    // Bounding box coordinates are already in image pixel coordinates from convertPredictionToDetections
    // Scale them to canvas coordinates
    const scaleX = ctx.canvas.width / videoWidth;
    const scaleY = ctx.canvas.height / videoHeight;
    
    let canvasX = boundingBox.x * scaleX;
    const canvasY = boundingBox.y * scaleY;
    const canvasBoxWidth = boundingBox.width * scaleX;
    const canvasBoxHeight = boundingBox.height * scaleY;
    
    // FIX: Camera is horizontally mirrored - flip X coordinates
    canvasX = ctx.canvas.width - canvasX - canvasBoxWidth;
    
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
