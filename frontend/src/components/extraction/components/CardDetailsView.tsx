import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ExtractedCard } from '../../../store/slices/cardExtractionSlice';
import { DetectionMetadata } from './DetectionMetadata';
import { ModelPlaceholders } from './ModelPlaceholders';
import { RotationControls } from './RotationControls';
import './CardDetailsView.css';

interface CardDetailsViewProps {
  card: ExtractedCard;
  cardIndex: number;
  totalCards: number;
  sourceImageDimensions?: { width: number; height: number };
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
  onBack,
  onCardUpdate,
  onNext,
  onPrevious
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'rotation' | 'models'>('overview');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [currentRotation, setCurrentRotation] = useState(0);
  const [tempRotation, setTempRotation] = useState(0);
  const [isProcessingRotation, setIsProcessingRotation] = useState(false);
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

    // Create temporary canvas for rotation
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Set original dimensions
    tempCanvas.width = card.imageData.width;
    tempCanvas.height = card.imageData.height;
    tempCtx.putImageData(card.imageData, 0, 0);

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
