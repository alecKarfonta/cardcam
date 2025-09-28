import React from 'react';
import { ExtractedCard } from '../../../store/slices/cardExtractionSlice';
import './DetectionMetadata.css';

interface DetectionMetadataProps {
  card: ExtractedCard;
  sourceImageDimensions?: { width: number; height: number };
}

export const DetectionMetadata: React.FC<DetectionMetadataProps> = ({
  card,
  sourceImageDimensions
}) => {
  const detection = card.originalDetection;
  
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatCoordinate = (value: number) => {
    return Math.round(value * 100) / 100;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return '#4CAF50'; // Green
    if (confidence >= 0.7) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  const calculateCardArea = () => {
    const { width, height } = detection.boundingBox;
    return Math.round(width * height);
  };

  const calculateRelativePosition = () => {
    if (!sourceImageDimensions) return null;
    
    const centerX = detection.boundingBox.x + detection.boundingBox.width / 2;
    const centerY = detection.boundingBox.y + detection.boundingBox.height / 2;
    
    const relativeX = (centerX / sourceImageDimensions.width) * 100;
    const relativeY = (centerY / sourceImageDimensions.height) * 100;
    
    return { x: relativeX, y: relativeY };
  };

  const getPositionDescription = () => {
    const position = calculateRelativePosition();
    if (!position) return 'Unknown';
    
    const { x, y } = position;
    let description = '';
    
    if (y < 33) description += 'Top ';
    else if (y > 66) description += 'Bottom ';
    else description += 'Middle ';
    
    if (x < 33) description += 'Left';
    else if (x > 66) description += 'Right';
    else description += 'Center';
    
    return description;
  };

  const relativePosition = calculateRelativePosition();

  return (
    <div className="detection-metadata">
      <div className="metadata-header">
        <h3>Detection Information</h3>
        <div 
          className="confidence-badge"
          style={{ backgroundColor: getConfidenceColor(detection.confidence) }}
        >
          {(detection.confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="metadata-sections">
        {/* Basic Detection Info */}
        <div className="metadata-section">
          <h4>Detection Details</h4>
          <div className="metadata-grid">
            <div className="metadata-item">
              <label>Detection ID:</label>
              <span className="monospace">{detection.id}</span>
            </div>
            <div className="metadata-item">
              <label>Confidence:</label>
              <span style={{ color: getConfidenceColor(detection.confidence) }}>
                {(detection.confidence * 100).toFixed(2)}%
              </span>
            </div>
            <div className="metadata-item">
              <label>Extracted At:</label>
              <span>{formatTimestamp(card.extractedAt || Date.now())}</span>
            </div>
            <div className="metadata-item">
              <label>Position:</label>
              <span>{getPositionDescription()}</span>
            </div>
          </div>
        </div>

        {/* Bounding Box Information */}
        <div className="metadata-section">
          <h4>Bounding Box</h4>
          <div className="metadata-grid">
            <div className="metadata-item">
              <label>X:</label>
              <span className="monospace">{formatCoordinate(detection.boundingBox.x)}px</span>
            </div>
            <div className="metadata-item">
              <label>Y:</label>
              <span className="monospace">{formatCoordinate(detection.boundingBox.y)}px</span>
            </div>
            <div className="metadata-item">
              <label>Width:</label>
              <span className="monospace">{formatCoordinate(detection.boundingBox.width)}px</span>
            </div>
            <div className="metadata-item">
              <label>Height:</label>
              <span className="monospace">{formatCoordinate(detection.boundingBox.height)}px</span>
            </div>
            <div className="metadata-item">
              <label>Area:</label>
              <span className="monospace">{calculateCardArea()}px²</span>
            </div>
            <div className="metadata-item">
              <label>Aspect Ratio:</label>
              <span className="monospace">
                {(detection.boundingBox.width / detection.boundingBox.height).toFixed(2)}:1
              </span>
            </div>
          </div>
        </div>

        {/* Relative Position */}
        {relativePosition && (
          <div className="metadata-section">
            <h4>Relative Position</h4>
            <div className="metadata-grid">
              <div className="metadata-item">
                <label>Center X:</label>
                <span className="monospace">{relativePosition.x.toFixed(1)}%</span>
              </div>
              <div className="metadata-item">
                <label>Center Y:</label>
                <span className="monospace">{relativePosition.y.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Oriented Bounding Box (if available) */}
        {detection.corners && (
          <div className="metadata-section">
            <h4>Oriented Bounding Box</h4>
            <div className="metadata-grid">
              <div className="metadata-item">
                <label>Rotation:</label>
                <span>{detection.isRotated ? 'Yes' : 'No'}</span>
              </div>
              <div className="metadata-item corners-list">
                <label>Corners:</label>
                <div className="corners-grid">
                  {detection.corners.map((corner, index) => (
                    <div key={index} className="corner-item">
                      <span className="corner-label">P{index + 1}:</span>
                      <span className="monospace">
                        ({formatCoordinate(corner.x)}, {formatCoordinate(corner.y)})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Image Dimensions */}
        <div className="metadata-section">
          <h4>Extracted Image</h4>
          <div className="metadata-grid">
            <div className="metadata-item">
              <label>Width:</label>
              <span className="monospace">
                {card.dimensions?.width || card.imageData.width}px
              </span>
            </div>
            <div className="metadata-item">
              <label>Height:</label>
              <span className="monospace">
                {card.dimensions?.height || card.imageData.height}px
              </span>
            </div>
            <div className="metadata-item">
              <label>Total Pixels:</label>
              <span className="monospace">
                {((card.dimensions?.width || card.imageData.width) * 
                  (card.dimensions?.height || card.imageData.height)).toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        {/* Source Image Context */}
        {sourceImageDimensions && (
          <div className="metadata-section">
            <h4>Source Image</h4>
            <div className="metadata-grid">
              <div className="metadata-item">
                <label>Source Width:</label>
                <span className="monospace">{sourceImageDimensions.width}px</span>
              </div>
              <div className="metadata-item">
                <label>Source Height:</label>
                <span className="monospace">{sourceImageDimensions.height}px</span>
              </div>
              <div className="metadata-item">
                <label>Card Coverage:</label>
                <span className="monospace">
                  {((calculateCardArea() / (sourceImageDimensions.width * sourceImageDimensions.height)) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Card Metadata (if available) */}
        {card.metadata && (
          <div className="metadata-section">
            <h4>Card Information</h4>
            <div className="metadata-grid">
              {card.metadata.cardName && (
                <div className="metadata-item">
                  <label>Card Name:</label>
                  <span>{card.metadata.cardName}</span>
                </div>
              )}
              {card.metadata.setName && (
                <div className="metadata-item">
                  <label>Set:</label>
                  <span>{card.metadata.setName}</span>
                </div>
              )}
              {card.metadata.rarity && (
                <div className="metadata-item">
                  <label>Rarity:</label>
                  <span>{card.metadata.rarity}</span>
                </div>
              )}
              {card.metadata.condition && (
                <div className="metadata-item">
                  <label>Condition:</label>
                  <span>{card.metadata.condition}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
