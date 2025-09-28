import React, { useState, useRef, useEffect } from 'react';
import { CardDetection } from '../../store/slices/inferenceSlice';
import './CardExtractionView.css';

export interface ExtractedCard {
  id: string;
  imageData: ImageData;
  originalDetection: CardDetection;
  canvas?: HTMLCanvasElement;
}

interface CardExtractionViewProps {
  extractedCards: ExtractedCard[];
  onBack: () => void;
  onCardSelect?: (card: ExtractedCard) => void;
}

export const CardExtractionView: React.FC<CardExtractionViewProps> = ({
  extractedCards,
  onBack,
  onCardSelect
}) => {
  const [selectedCardIndex, setSelectedCardIndex] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.0);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  // Filter cards based on confidence threshold
  const filteredCards = extractedCards.filter(
    card => card.originalDetection.confidence >= confidenceThreshold
  );

  // Reset selected card index if it's no longer valid after filtering
  useEffect(() => {
    if (selectedCardIndex !== null && selectedCardIndex >= filteredCards.length) {
      setSelectedCardIndex(null);
    }
  }, [filteredCards.length, selectedCardIndex]);

  // Render extracted cards to canvases
  useEffect(() => {
    extractedCards.forEach((card, index) => {
      const canvas = canvasRefs.current[index];
      if (canvas && card.imageData) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          canvas.width = card.imageData.width;
          canvas.height = card.imageData.height;
          ctx.putImageData(card.imageData, 0, 0);
        }
      }
    });
  }, [extractedCards]);

  const handleCardClick = (index: number) => {
    setSelectedCardIndex(index);
    if (onCardSelect) {
      onCardSelect(extractedCards[index]);
    }
  };

  const handleDownloadCard = (card: ExtractedCard, index: number) => {
    const canvas = canvasRefs.current[index];
    if (canvas) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `extracted-card-${index + 1}-${timestamp}.png`;
      
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
    }
  };

  const handleDownloadAll = () => {
    extractedCards.forEach((card, index) => {
      setTimeout(() => handleDownloadCard(card, index), index * 100);
    });
  };

  const renderCardGrid = () => {
    if (filteredCards.length === 0) {
      return (
        <div className="no-cards-message">
          <p>No cards match the current confidence threshold ({(confidenceThreshold * 100).toFixed(0)}%)</p>
          <p>Try lowering the threshold to see more cards.</p>
        </div>
      );
    }

    return (
      <div className="card-grid">
        {filteredCards.map((card, index) => (
          <div
            key={card.id}
            className={`card-item ${selectedCardIndex === index ? 'selected' : ''}`}
            onClick={() => handleCardClick(index)}
          >
            <div className="card-preview">
              <canvas
                ref={(el) => (canvasRefs.current[index] = el)}
                className="card-canvas"
              />
              <div className="card-overlay">
                <div className="card-info">
                  <span className="card-confidence">
                    {(card.originalDetection.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="card-dimensions">
                    {card.imageData.width}×{card.imageData.height}
                  </span>
                </div>
                <button
                  className="download-card-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownloadCard(card, index);
                  }}
                >
                  ⬇
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderDetailView = () => {
    if (selectedCardIndex === null) return null;
    
    const selectedCard = filteredCards[selectedCardIndex];
    
    return (
      <div className="card-detail-view">
        <div className="detail-header">
          <button
            className="back-to-grid-btn"
            onClick={() => setSelectedCardIndex(null)}
          >
            ← Back to Grid
          </button>
          <div className="detail-info">
            <span>Card {selectedCardIndex + 1} of {filteredCards.length}</span>
            <span>Confidence: {(selectedCard.originalDetection.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
        
        <div className="detail-canvas-container">
          <div 
            className="detail-canvas-wrapper"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            <canvas
              ref={(el) => (canvasRefs.current[selectedCardIndex] = el)}
              className="detail-canvas"
            />
          </div>
        </div>
        
        <div className="detail-controls">
          <div className="zoom-controls">
            <button
              className="zoom-btn"
              onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.25))}
              disabled={zoomLevel <= 0.5}
            >
              -
            </button>
            <span className="zoom-level">{Math.round(zoomLevel * 100)}%</span>
            <button
              className="zoom-btn"
              onClick={() => setZoomLevel(Math.min(4, zoomLevel + 0.25))}
              disabled={zoomLevel >= 4}
            >
              +
            </button>
          </div>
          
          <button
            className="download-btn"
            onClick={() => handleDownloadCard(selectedCard, selectedCardIndex)}
          >
            Download Card
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="card-extraction-view">
      <div className="extraction-header">
        <button className="back-btn" onClick={onBack}>
          ← Back to Camera
        </button>
        <div className="extraction-title">
          <h2>Extracted Cards ({filteredCards.length}/{extractedCards.length})</h2>
        </div>
        <div className="threshold-controls">
          <label className="threshold-label">
            Confidence: {(confidenceThreshold * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            className="threshold-slider"
          />
        </div>
        <button className="download-all-btn" onClick={handleDownloadAll}>
          Download All
        </button>
      </div>

      <div className="extraction-content">
        {selectedCardIndex !== null ? renderDetailView() : renderCardGrid()}
      </div>
    </div>
  );
};
