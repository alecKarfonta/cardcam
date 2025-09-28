import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ExtractedCard } from '../../../store/slices/cardExtractionSlice';
import { DetectionMetadata } from './DetectionMetadata';
import { ModelPlaceholders } from './ModelPlaceholders';
import { RotationControls } from './RotationControls';
import { DimensionControls, DimensionAdjustments } from './DimensionControls';
import { CardReExtractor } from '../../../utils/CardReExtractor';
import './CardDetailsView.css';

interface CardDetailsViewProps {
  card: ExtractedCard;
  cardIndex: number;
  totalCards: number;
  sourceImageDimensions?: { width: number; height: number };
  sourceImageData?: ImageData; // Original image for re-extraction
  onBack: () => void;
  onCardUpdate?: (updatedCard: ExtractedCard) => void;
  onNext?: () => void;
  onPrevious?: () => void;
}

export const CardDetailsView: React.FC<CardDetailsViewProps> = ({
  card,
  cardIndex,
  totalCards,
  sourceImageDimensions,
  sourceImageData,
  onBack,
  onCardUpdate,
  onNext,
  onPrevious
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'rotation' | 'dimensions' | 'models'>('overview');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [currentRotation, setCurrentRotation] = useState(0);
  const [tempRotation, setTempRotation] = useState(0);
  const [isProcessingRotation, setIsProcessingRotation] = useState(false);
  const [isProcessingDimensions, setIsProcessingDimensions] = useState(false);
  const [currentDimensionAdjustments, setCurrentDimensionAdjustments] = useState<DimensionAdjustments>({
    top: 0, bottom: 0, left: 0, right: 0
  });
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Render the card image to canvas with current rotation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !card.imageData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // DEBUG: Check ImageData content
    console.log(`🖼️ CardDetailsView DEBUG: Rendering card ${card.id}`);
    console.log(`🖼️ CardDetailsView DEBUG: ImageData dimensions: ${card.imageData.width}x${card.imageData.height}`);
    console.log(`🖼️ CardDetailsView DEBUG: ImageData length: ${card.imageData.data.length}`);
    
    // Check if ImageData is blank
    let nonZeroPixels = 0;
    for (let i = 0; i < card.imageData.data.length; i += 4) {
      if (card.imageData.data[i] !== 0 || card.imageData.data[i + 1] !== 0 || card.imageData.data[i + 2] !== 0 || card.imageData.data[i + 3] !== 0) {
        nonZeroPixels++;
      }
    }
    const totalPixels = card.imageData.data.length / 4;
    const nonZeroPercentage = (nonZeroPixels / totalPixels) * 100;
    console.log(`🖼️ CardDetailsView DEBUG: Non-zero pixels: ${nonZeroPixels}/${totalPixels} (${nonZeroPercentage.toFixed(1)}%)`);
    
    if (nonZeroPixels === 0) {
      console.error(`❌ CardDetailsView DEBUG: ImageData is completely blank for card ${card.id}!`);
    } else if (nonZeroPercentage < 1) {
      console.warn(`⚠️ CardDetailsView DEBUG: ImageData has very few non-zero pixels (${nonZeroPercentage.toFixed(1)}%) for card ${card.id}`);
    }

    // Create temporary canvas for rotation
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Set original dimensions
    tempCanvas.width = card.imageData.width;
    tempCanvas.height = card.imageData.height;
    tempCtx.putImageData(card.imageData, 0, 0);
    
    console.log(`🖼️ CardDetailsView DEBUG: Created temp canvas ${tempCanvas.width}x${tempCanvas.height}`);

    // Calculate rotated dimensions
    const radians = (tempRotation * Math.PI) / 180;
    const cos = Math.abs(Math.cos(radians));
    const sin = Math.abs(Math.sin(radians));
    const newWidth = card.imageData.width * cos + card.imageData.height * sin;
    const newHeight = card.imageData.width * sin + card.imageData.height * cos;

    // Set canvas dimensions
    canvas.width = newWidth;
    canvas.height = newHeight;

    // Clear and apply rotation
    ctx.clearRect(0, 0, newWidth, newHeight);
    ctx.save();
    ctx.translate(newWidth / 2, newHeight / 2);
    ctx.rotate(radians);
    ctx.drawImage(tempCanvas, -card.imageData.width / 2, -card.imageData.height / 2);
    ctx.restore();
  }, [card.imageData, tempRotation]);

  const handleRotationChange = useCallback((rotation: number) => {
    setTempRotation(rotation);
  }, []);

  const handleApplyRotation = useCallback(async () => {
    setIsProcessingRotation(true);
    
    try {
      // In a real implementation, this would apply the rotation to the actual image data
      // For now, we'll simulate the process
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setCurrentRotation(tempRotation);
      
      if (onCardUpdate) {
        // Create updated card with rotation metadata
        const updatedCard: ExtractedCard = {
          ...card,
          metadata: {
            ...card.metadata,
            // Add rotation info to metadata
          }
        };
        onCardUpdate(updatedCard);
      }
    } catch (error) {
      console.error('Failed to apply rotation:', error);
    } finally {
      setIsProcessingRotation(false);
    }
  }, [tempRotation, card, onCardUpdate]);

  const handleResetRotation = useCallback(() => {
    setTempRotation(0);
    setCurrentRotation(0);
  }, []);

  const handleZoomChange = useCallback((delta: number) => {
    setZoomLevel(prev => Math.max(0.25, Math.min(4, prev + delta)));
  }, []);

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (event.button === 0) { // Left mouse button
      setIsDragging(true);
      setDragStart({ x: event.clientX - panOffset.x, y: event.clientY - panOffset.y });
    }
  }, [panOffset]);

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (isDragging) {
      setPanOffset({
        x: event.clientX - dragStart.x,
        y: event.clientY - dragStart.y
      });
    }
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Touch event handlers for mobile
  const handleTouchStart = useCallback((event: React.TouchEvent) => {
    if (event.touches.length === 1) {
      const touch = event.touches[0];
      setIsDragging(true);
      setDragStart({ x: touch.clientX - panOffset.x, y: touch.clientY - panOffset.y });
    }
  }, [panOffset]);

  const handleTouchMove = useCallback((event: React.TouchEvent) => {
    if (isDragging && event.touches.length === 1) {
      event.preventDefault(); // Prevent scrolling
      const touch = event.touches[0];
      setPanOffset({
        x: touch.clientX - dragStart.x,
        y: touch.clientY - dragStart.y
      });
    }
  }, [isDragging, dragStart]);

  const handleTouchEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleWheel = useCallback((event: React.WheelEvent) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.1 : 0.1;
    handleZoomChange(delta);
  }, [handleZoomChange]);

  const handleDownload = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `card-${cardIndex + 1}-${timestamp}.png`;
    
    canvas.toBlob((blob) => {
      if (blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    }, 'image/png');
  }, [cardIndex]);

  const handleModelResultUpdate = useCallback((cardId: string, modelType: string, result: any) => {
    console.log('Model result update:', { cardId, modelType, result });
    // In a real implementation, this would update the card with model results
    if (onCardUpdate) {
      const updatedCard: ExtractedCard = {
        ...card,
        metadata: {
          ...card.metadata,
          // Add model results to metadata
        }
      };
      onCardUpdate(updatedCard);
    }
  }, [card, onCardUpdate]);

  const handleDimensionChange = useCallback((adjustments: DimensionAdjustments) => {
    setCurrentDimensionAdjustments(adjustments);
  }, []);

  const handleReExtract = useCallback(async (adjustments: DimensionAdjustments) => {
    if (!sourceImageData) {
      console.error('❌ No source image data available for re-extraction');
      return;
    }

    if (!card.extractionMetadata) {
      console.error('❌ No extraction metadata available for re-extraction');
      return;
    }

    setIsProcessingDimensions(true);

    try {
      console.log('🔄 Starting re-extraction with adjustments:', adjustments);
      
      // Validate adjustments
      const validation = CardReExtractor.validateAdjustments(
        card.dimensions,
        adjustments,
        card.extractionMetadata.originalSize
      );

      if (!validation.valid) {
        console.error('❌ Invalid adjustments:', validation.reason);
        alert(`Cannot apply adjustments: ${validation.reason}`);
        return;
      }

      // Perform re-extraction
      const reExtractedCard = await CardReExtractor.reExtractCard(
        sourceImageData,
        card,
        adjustments
      );

      if (reExtractedCard && onCardUpdate) {
        console.log('✅ Re-extraction successful, updating card');
        onCardUpdate(reExtractedCard);
        
        // Reset adjustments after successful re-extraction
        setCurrentDimensionAdjustments({ top: 0, bottom: 0, left: 0, right: 0 });
      } else {
        console.error('❌ Re-extraction failed');
        alert('Re-extraction failed. Please try different adjustments.');
      }

    } catch (error) {
      console.error('❌ Re-extraction error:', error);
      alert('An error occurred during re-extraction. Please try again.');
    } finally {
      setIsProcessingDimensions(false);
    }
  }, [sourceImageData, card, onCardUpdate]);

  const renderImageViewer = () => (
    <div className="image-viewer-container">
      <div className="viewer-header">
        <div className="viewer-info">
          <h3>Card {cardIndex + 1} of {totalCards}</h3>
          <span className="confidence-badge" style={{ 
            backgroundColor: card.originalDetection.confidence >= 0.9 ? '#4CAF50' : 
                           card.originalDetection.confidence >= 0.7 ? '#FF9800' : '#F44336'
          }}>
            {(card.originalDetection.confidence * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="viewer-controls">
          <div className="zoom-controls">
            <button 
              className="zoom-btn"
              onClick={() => handleZoomChange(-0.25)}
              disabled={zoomLevel <= 0.25}
            >
              -
            </button>
            <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
            <button 
              className="zoom-btn"
              onClick={() => handleZoomChange(0.25)}
              disabled={zoomLevel >= 4}
            >
              +
            </button>
          </div>
          
          <button className="reset-view-btn" onClick={() => {
            setZoomLevel(1);
            setPanOffset({ x: 0, y: 0 });
          }}>
            Reset View
          </button>
          
          <button className="download-btn" onClick={handleDownload}>
            Download
          </button>
        </div>
      </div>
      
      <div 
        className="canvas-container"
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onWheel={handleWheel}
      >
        <div 
          className="canvas-wrapper"
          style={{ 
            transform: `scale(${zoomLevel}) translate(${panOffset.x / zoomLevel}px, ${panOffset.y / zoomLevel}px)`,
            cursor: isDragging ? 'grabbing' : 'grab'
          }}
        >
          <canvas
            ref={canvasRef}
            className="card-canvas"
          />
        </div>
      </div>
      
      <div className="viewer-footer">
        <div className="navigation-controls">
          <button 
            className="nav-btn"
            onClick={onPrevious}
            disabled={!onPrevious || cardIndex === 0}
          >
            ← Previous
          </button>
          <button 
            className="nav-btn"
            onClick={onNext}
            disabled={!onNext || cardIndex === totalCards - 1}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="tab-content">
            <DetectionMetadata 
              card={card} 
              sourceImageDimensions={sourceImageDimensions}
            />
          </div>
        );
      
      case 'rotation':
        return (
          <div className="tab-content">
            <RotationControls
              currentRotation={currentRotation}
              onRotationChange={handleRotationChange}
              onApplyRotation={handleApplyRotation}
              onResetRotation={handleResetRotation}
              isProcessing={isProcessingRotation}
            />
          </div>
        );
      
      case 'dimensions':
        return (
          <div className="tab-content">
            <DimensionControls
              card={card}
              onDimensionChange={handleDimensionChange}
              onReExtract={handleReExtract}
              isProcessing={isProcessingDimensions}
            />
          </div>
        );
      
      case 'models':
        return (
          <div className="tab-content">
            <ModelPlaceholders
              cardId={card.id}
              onModelResultUpdate={handleModelResultUpdate}
            />
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="card-details-view">
      <div className="details-header">
        <button className="back-btn" onClick={onBack}>
          ← Back to Grid
        </button>
        <h2>Card Details</h2>
      </div>

      <div className="details-content">
        <div className="image-section">
          {renderImageViewer()}
        </div>
        
        <div className="info-section">
          <div className="tab-navigation">
            <button 
              className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button 
              className={`tab-btn ${activeTab === 'rotation' ? 'active' : ''}`}
              onClick={() => setActiveTab('rotation')}
            >
              Rotation
            </button>
            <button 
              className={`tab-btn ${activeTab === 'dimensions' ? 'active' : ''}`}
              onClick={() => setActiveTab('dimensions')}
              disabled={!sourceImageData}
              title={!sourceImageData ? 'Original image data not available' : 'Adjust extraction dimensions'}
            >
              Dimensions
            </button>
            <button 
              className={`tab-btn ${activeTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveTab('models')}
            >
              AI Models
            </button>
          </div>
          
          <div className="tab-content-container">
            {renderTabContent()}
          </div>
        </div>
      </div>
    </div>
  );
};
