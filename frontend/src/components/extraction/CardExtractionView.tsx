import React, { useState, useRef, useEffect } from 'react';
import { ExtractedCard } from '../../store/slices/cardExtractionSlice';
import { CardDetailsView } from './components/CardDetailsView';
import './CardExtractionView.css';

interface CardExtractionViewProps {
  extractedCards: ExtractedCard[];
  sourceImageDimensions?: { width: number; height: number };
  onBack: () => void;
  onCardSelect?: (card: ExtractedCard) => void;
  onCardUpdate?: (updatedCard: ExtractedCard) => void;
}

export const CardExtractionView: React.FC<CardExtractionViewProps> = ({
  extractedCards,
  sourceImageDimensions,
  onBack,
  onCardSelect,
  onCardUpdate
}) => {
  const [selectedCardIndex, setSelectedCardIndex] = useState<number | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.0);
  const [viewMode, setViewMode] = useState<'grid' | 'details'>('grid');
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  // Filter cards based on confidence threshold
  const filteredCards = extractedCards.filter(
    card => card.originalDetection.confidence >= confidenceThreshold
  );

  // Reset selected card index if it's no longer valid after filtering
  useEffect(() => {
    if (selectedCardIndex !== null && selectedCardIndex >= filteredCards.length) {
      setSelectedCardIndex(null);
      setViewMode('grid');
    }
  }, [filteredCards.length, selectedCardIndex]);

  // Render extracted cards to canvases
  useEffect(() => {
    const renderCanvases = () => {
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
    };

    // Always render canvases, but with delay when returning to grid
    if (viewMode === 'grid') {
      // Use requestAnimationFrame for better timing when returning to grid
      requestAnimationFrame(() => {
        setTimeout(renderCanvases, 50); // Increased delay to ensure DOM is ready
      });
    } else {
      renderCanvases();
    }
  }, [extractedCards, viewMode]);

  // Force re-render grid canvases when returning to grid view
  useEffect(() => {
    if (viewMode === 'grid') {
      console.log('🎨 Force re-rendering grid canvases');
      // Additional effect to ensure grid canvases are rendered when switching back
      const timer = setTimeout(() => {
        let renderedCount = 0;
        extractedCards.forEach((card, index) => {
          const canvas = canvasRefs.current[index];
          if (canvas && card.imageData) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
              canvas.width = card.imageData.width;
              canvas.height = card.imageData.height;
              ctx.putImageData(card.imageData, 0, 0);
              renderedCount++;
            }
          }
        });
        console.log(`✅ Rendered ${renderedCount}/${extractedCards.length} canvases`);
      }, 100); // Longer delay to ensure all DOM updates are complete

      return () => clearTimeout(timer);
    }
  }, [viewMode, extractedCards]);

  const handleCardClick = (index: number) => {
    setSelectedCardIndex(index);
    setViewMode('details');
    if (onCardSelect) {
      onCardSelect(filteredCards[index]);
    }
  };

  const handleBackToGrid = () => {
    console.log('🔄 Switching back to grid view');
    console.log('📊 Canvas refs status:', canvasRefs.current.map((ref, i) => ({ index: i, hasRef: !!ref, cardId: extractedCards[i]?.id })));
    setSelectedCardIndex(null);
    setViewMode('grid');
  };

  const handleNextCard = () => {
    if (selectedCardIndex !== null && selectedCardIndex < filteredCards.length - 1) {
      setSelectedCardIndex(selectedCardIndex + 1);
    }
  };

  const handlePreviousCard = () => {
    if (selectedCardIndex !== null && selectedCardIndex > 0) {
      setSelectedCardIndex(selectedCardIndex - 1);
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
                ref={(el) => {
                  const originalIndex = extractedCards.findIndex(c => c.id === card.id);
                  canvasRefs.current[originalIndex] = el;
                  // Immediately render if canvas and imageData are available
                  if (el && card.imageData) {
                    const ctx = el.getContext('2d');
                    if (ctx) {
                      el.width = card.imageData.width;
                      el.height = card.imageData.height;
                      ctx.putImageData(card.imageData, 0, 0);
                    }
                  }
                }}
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
                    handleDownloadCard(card, extractedCards.findIndex(c => c.id === card.id));
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

  // Show the enhanced card details view when a card is selected
  if (viewMode === 'details' && selectedCardIndex !== null) {
    const selectedCard = filteredCards[selectedCardIndex];
    return (
      <CardDetailsView
        card={selectedCard}
        cardIndex={selectedCardIndex}
        totalCards={filteredCards.length}
        sourceImageDimensions={sourceImageDimensions}
        onBack={handleBackToGrid}
        onCardUpdate={onCardUpdate}
        onNext={handleNextCard}
        onPrevious={handlePreviousCard}
      />
    );
  }

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
        {renderCardGrid()}
      </div>
    </div>
  );
};
