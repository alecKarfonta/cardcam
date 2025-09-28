import React, { useState, useCallback } from 'react';
import { ExtractedCard } from '../../../store/slices/cardExtractionSlice';
import './DimensionControls.css';

interface DimensionControlsProps {
  card: ExtractedCard;
  onDimensionChange: (newDimensions: DimensionAdjustments) => void;
  onReExtract: (adjustments: DimensionAdjustments) => void;
  isProcessing?: boolean;
}

export interface DimensionAdjustments {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

export const DimensionControls: React.FC<DimensionControlsProps> = ({
  card,
  onDimensionChange,
  onReExtract,
  isProcessing = false
}) => {
  const [adjustments, setAdjustments] = useState<DimensionAdjustments>({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0
  });

  const [stepSize, setStepSize] = useState(20); // Default step size in pixels

  const handleAdjustment = useCallback((side: keyof DimensionAdjustments, delta: number) => {
    const newAdjustments = {
      ...adjustments,
      [side]: Math.max(-100, Math.min(200, adjustments[side] + delta)) // Limit between -100 and +200 pixels
    };
    setAdjustments(newAdjustments);
    onDimensionChange(newAdjustments);
  }, [adjustments, onDimensionChange]);

  const handleReset = useCallback(() => {
    const resetAdjustments = { top: 0, bottom: 0, left: 0, right: 0 };
    setAdjustments(resetAdjustments);
    onDimensionChange(resetAdjustments);
  }, [onDimensionChange]);

  const handleReExtract = useCallback(() => {
    onReExtract(adjustments);
  }, [adjustments, onReExtract]);

  const hasAdjustments = Object.values(adjustments).some(val => val !== 0);

  // Get current extraction info
  const extractionMeta = card.extractionMetadata;
  const originalDetection = card.originalDetection;

  return (
    <div className="dimension-controls">
      <div className="controls-header">
        <h3>Adjust Extraction Dimensions</h3>
        <p className="controls-description">
          Extend or reduce the extraction area to get more or less of the original image
        </p>
      </div>

      <div className="current-info">
        <div className="info-item">
          <span className="info-label">Current Size:</span>
          <span className="info-value">{card.dimensions.width} × {card.dimensions.height}px</span>
        </div>
        {extractionMeta && (
          <div className="info-item">
            <span className="info-label">Original Image:</span>
            <span className="info-value">{extractionMeta.originalSize.width} × {extractionMeta.originalSize.height}px</span>
          </div>
        )}
        <div className="info-item">
          <span className="info-label">Confidence:</span>
          <span className="info-value">{(originalDetection.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="step-size-control">
        <label htmlFor="step-size">Step Size:</label>
        <select 
          id="step-size" 
          value={stepSize} 
          onChange={(e) => setStepSize(parseInt(e.target.value))}
          className="step-size-select"
        >
          <option value={10}>10px</option>
          <option value={20}>20px</option>
          <option value={50}>50px</option>
          <option value={100}>100px</option>
        </select>
      </div>

      <div className="dimension-grid">
        {/* Top controls */}
        <div className="control-section top-section">
          <div className="control-group">
            <button 
              className="dimension-btn corner-btn top-left"
              onClick={() => {
                handleAdjustment('top', stepSize);
                handleAdjustment('left', stepSize);
              }}
              disabled={isProcessing}
              title="Extend top-left corner"
            >
              ↖
            </button>
            
            <div className="side-controls">
              <button 
                className="dimension-btn side-btn"
                onClick={() => handleAdjustment('top', -stepSize)}
                disabled={isProcessing}
                title={`Reduce top by ${stepSize}px`}
              >
                ▼
              </button>
              <span className="adjustment-value">{adjustments.top > 0 ? '+' : ''}{adjustments.top}px</span>
              <button 
                className="dimension-btn side-btn"
                onClick={() => handleAdjustment('top', stepSize)}
                disabled={isProcessing}
                title={`Extend top by ${stepSize}px`}
              >
                ▲
              </button>
            </div>

            <button 
              className="dimension-btn corner-btn top-right"
              onClick={() => {
                handleAdjustment('top', stepSize);
                handleAdjustment('right', stepSize);
              }}
              disabled={isProcessing}
              title="Extend top-right corner"
            >
              ↗
            </button>
          </div>
        </div>

        {/* Middle row with left and right controls */}
        <div className="control-section middle-section">
          <div className="side-controls left-controls">
            <button 
              className="dimension-btn side-btn"
              onClick={() => handleAdjustment('left', -stepSize)}
              disabled={isProcessing}
              title={`Reduce left by ${stepSize}px`}
            >
              ▶
            </button>
            <span className="adjustment-value">{adjustments.left > 0 ? '+' : ''}{adjustments.left}px</span>
            <button 
              className="dimension-btn side-btn"
              onClick={() => handleAdjustment('left', stepSize)}
              disabled={isProcessing}
              title={`Extend left by ${stepSize}px`}
            >
              ◀
            </button>
          </div>

          <div className="card-preview-area">
            <div className="preview-box">
              <span className="preview-label">Card Preview Area</span>
              <div className="current-dimensions">
                {card.dimensions.width} × {card.dimensions.height}
              </div>
            </div>
          </div>

          <div className="side-controls right-controls">
            <button 
              className="dimension-btn side-btn"
              onClick={() => handleAdjustment('right', -stepSize)}
              disabled={isProcessing}
              title={`Reduce right by ${stepSize}px`}
            >
              ◀
            </button>
            <span className="adjustment-value">{adjustments.right > 0 ? '+' : ''}{adjustments.right}px</span>
            <button 
              className="dimension-btn side-btn"
              onClick={() => handleAdjustment('right', stepSize)}
              disabled={isProcessing}
              title={`Extend right by ${stepSize}px`}
            >
              ▶
            </button>
          </div>
        </div>

        {/* Bottom controls */}
        <div className="control-section bottom-section">
          <div className="control-group">
            <button 
              className="dimension-btn corner-btn bottom-left"
              onClick={() => {
                handleAdjustment('bottom', stepSize);
                handleAdjustment('left', stepSize);
              }}
              disabled={isProcessing}
              title="Extend bottom-left corner"
            >
              ↙
            </button>
            
            <div className="side-controls">
              <button 
                className="dimension-btn side-btn"
                onClick={() => handleAdjustment('bottom', -stepSize)}
                disabled={isProcessing}
                title={`Reduce bottom by ${stepSize}px`}
              >
                ▲
              </button>
              <span className="adjustment-value">{adjustments.bottom > 0 ? '+' : ''}{adjustments.bottom}px</span>
              <button 
                className="dimension-btn side-btn"
                onClick={() => handleAdjustment('bottom', stepSize)}
                disabled={isProcessing}
                title={`Extend bottom by ${stepSize}px`}
              >
                ▼
              </button>
            </div>

            <button 
              className="dimension-btn corner-btn bottom-right"
              onClick={() => {
                handleAdjustment('bottom', stepSize);
                handleAdjustment('right', stepSize);
              }}
              disabled={isProcessing}
              title="Extend bottom-right corner"
            >
              ↘
            </button>
          </div>
        </div>
      </div>

      <div className="action-buttons">
        <button 
          className="reset-btn"
          onClick={handleReset}
          disabled={isProcessing || !hasAdjustments}
        >
          Reset
        </button>
        
        <button 
          className="re-extract-btn"
          onClick={handleReExtract}
          disabled={isProcessing || !hasAdjustments}
        >
          {isProcessing ? 'Re-extracting...' : 'Apply Changes'}
        </button>
      </div>

      {hasAdjustments && (
        <div className="adjustment-summary">
          <h4>Planned Changes:</h4>
          <ul>
            {adjustments.top !== 0 && <li>Top: {adjustments.top > 0 ? '+' : ''}{adjustments.top}px</li>}
            {adjustments.bottom !== 0 && <li>Bottom: {adjustments.bottom > 0 ? '+' : ''}{adjustments.bottom}px</li>}
            {adjustments.left !== 0 && <li>Left: {adjustments.left > 0 ? '+' : ''}{adjustments.left}px</li>}
            {adjustments.right !== 0 && <li>Right: {adjustments.right > 0 ? '+' : ''}{adjustments.right}px</li>}
          </ul>
          <p className="new-dimensions">
            New size: {card.dimensions.width + adjustments.left + adjustments.right} × {card.dimensions.height + adjustments.top + adjustments.bottom}px
          </p>
        </div>
      )}
    </div>
  );
};
