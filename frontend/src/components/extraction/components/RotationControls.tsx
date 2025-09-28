import React, { useState, useCallback } from 'react';
import './RotationControls.css';

interface RotationControlsProps {
  currentRotation: number;
  onRotationChange: (rotation: number) => void;
  onApplyRotation: () => void;
  onResetRotation: () => void;
  isProcessing?: boolean;
}

export const RotationControls: React.FC<RotationControlsProps> = ({
  currentRotation,
  onRotationChange,
  onApplyRotation,
  onResetRotation,
  isProcessing = false
}) => {
  const [tempRotation, setTempRotation] = useState(currentRotation);

  const handleSliderChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newRotation = parseInt(event.target.value);
    setTempRotation(newRotation);
    onRotationChange(newRotation);
  }, [onRotationChange]);

  const handlePresetRotation = useCallback((degrees: number) => {
    const newRotation = (currentRotation + degrees) % 360;
    const normalizedRotation = newRotation < 0 ? newRotation + 360 : newRotation;
    setTempRotation(normalizedRotation);
    onRotationChange(normalizedRotation);
  }, [currentRotation, onRotationChange]);

  const handleReset = useCallback(() => {
    setTempRotation(0);
    onResetRotation();
  }, [onResetRotation]);

  const formatRotation = (rotation: number) => {
    return `${rotation}°`;
  };

  const getRotationDescription = (rotation: number) => {
    if (rotation === 0) return 'Original orientation';
    if (rotation === 90) return 'Rotated 90° clockwise';
    if (rotation === 180) return 'Rotated 180°';
    if (rotation === 270) return 'Rotated 90° counter-clockwise';
    return `Rotated ${rotation}°`;
  };

  return (
    <div className="rotation-controls">
      <div className="controls-header">
        <h4>Rotation Correction</h4>
        <div className="rotation-display">
          <span className="rotation-value">{formatRotation(tempRotation)}</span>
          <span className="rotation-description">{getRotationDescription(tempRotation)}</span>
        </div>
      </div>

      <div className="controls-content">
        {/* Rotation Slider */}
        <div className="slider-section">
          <label className="slider-label">Fine Adjustment</label>
          <div className="slider-container">
            <span className="slider-min">0°</span>
            <input
              type="range"
              min="0"
              max="359"
              value={tempRotation}
              onChange={handleSliderChange}
              className="rotation-slider"
              disabled={isProcessing}
            />
            <span className="slider-max">359°</span>
          </div>
          <div className="slider-markers">
            <div className="marker" style={{ left: '0%' }}>0°</div>
            <div className="marker" style={{ left: '25%' }}>90°</div>
            <div className="marker" style={{ left: '50%' }}>180°</div>
            <div className="marker" style={{ left: '75%' }}>270°</div>
          </div>
        </div>

        {/* Preset Rotation Buttons */}
        <div className="preset-section">
          <label className="preset-label">Quick Rotations</label>
          <div className="preset-buttons">
            <button
              className="preset-btn rotate-left"
              onClick={() => handlePresetRotation(-90)}
              disabled={isProcessing}
              title="Rotate 90° counter-clockwise"
            >
              <svg viewBox="0 0 24 24" className="rotate-icon">
                <path d="M7.11 8.53L5.7 7.11C4.8 8.27 4.24 9.61 4.07 11h2.02c.14-.87.49-1.72 1.02-2.47zM6.09 13H4.07c.17 1.39.72 2.73 1.62 3.89l1.41-1.42c-.52-.75-.87-1.59-1.01-2.47zm1.01 5.32c1.16.9 2.51 1.44 3.9 1.61V17.91c-.87-.15-1.71-.49-2.46-1.03L7.1 18.32zM13 4.07V1L8.45 5.55 13 10V6.09c2.84.48 5 2.94 5 5.91s-2.16 5.43-5 5.91v2.02c3.95-.49 7-3.85 7-7.93s-3.05-7.44-7-7.93z"/>
              </svg>
              90° CCW
            </button>
            
            <button
              className="preset-btn rotate-180"
              onClick={() => handlePresetRotation(180)}
              disabled={isProcessing}
              title="Rotate 180°"
            >
              <svg viewBox="0 0 24 24" className="rotate-icon">
                <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
              </svg>
              180°
            </button>
            
            <button
              className="preset-btn rotate-right"
              onClick={() => handlePresetRotation(90)}
              disabled={isProcessing}
              title="Rotate 90° clockwise"
            >
              <svg viewBox="0 0 24 24" className="rotate-icon">
                <path d="M16.89 15.5l1.42 1.41c.9-1.16 1.45-2.5 1.62-3.91h-2.02c-.14.87-.48 1.72-1.02 2.5zM13 7.83V5.7c-.84-.15-1.69-.22-2.55-.22-1.52 0-2.98.29-4.34.82L7.52 7.71c.84-.32 1.74-.54 2.68-.62.28-.02.57-.04.85-.04.28 0 .56.01.84.02V9.5L15 5 10 .5v2.17C6.06.91 2.91 4.27 2.91 8.34s3.05 7.44 7 7.93v-2.02c-2.84-.48-5-2.94-5-5.91s2.16-5.43 5-5.91z"/>
              </svg>
              90° CW
            </button>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="action-section">
          <div className="action-buttons">
            <button
              className="action-btn reset-btn"
              onClick={handleReset}
              disabled={isProcessing || tempRotation === 0}
            >
              <svg viewBox="0 0 24 24" className="action-icon">
                <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
              </svg>
              Reset
            </button>
            
            <button
              className="action-btn apply-btn"
              onClick={onApplyRotation}
              disabled={isProcessing || tempRotation === currentRotation}
            >
              {isProcessing ? (
                <>
                  <div className="spinner"></div>
                  Processing...
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" className="action-icon">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                  </svg>
                  Apply Rotation
                </>
              )}
            </button>
          </div>
        </div>

        {/* Rotation Tips */}
        <div className="tips-section">
          <div className="tips-content">
            <h5>Tips:</h5>
            <ul>
              <li>Use the slider for precise angle adjustments</li>
              <li>Quick buttons rotate in 90° increments</li>
              <li>Preview shows real-time rotation changes</li>
              <li>Apply button saves the rotation permanently</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
