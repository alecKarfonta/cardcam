import React, { useState } from 'react';
import './ModelPlaceholders.css';

interface ModelPlaceholdersProps {
  cardId: string;
  onModelResultUpdate?: (cardId: string, modelType: string, result: any) => void;
}

interface CardTypeResult {
  type: 'pokemon' | 'magic' | 'yugioh' | 'sports' | 'other';
  confidence: number;
  subtype?: string;
}

interface OCRResult {
  extractedText: string;
  confidence: number;
  textRegions: Array<{
    text: string;
    confidence: number;
    boundingBox: { x: number; y: number; width: number; height: number };
  }>;
}

interface CardIdentificationResult {
  cardName: string;
  setName: string;
  setCode: string;
  rarity: string;
  cardNumber: string;
  confidence: number;
  estimatedValue?: {
    low: number;
    market: number;
    high: number;
    currency: string;
  };
}

export const ModelPlaceholders: React.FC<ModelPlaceholdersProps> = ({
  cardId,
  onModelResultUpdate
}) => {
  const [activeTab, setActiveTab] = useState<'cardType' | 'ocr' | 'identification'>('cardType');
  const [isProcessing, setIsProcessing] = useState<Record<string, boolean>>({});

  // Mock data for demonstration
  const mockCardTypeResult: CardTypeResult = {
    type: 'pokemon',
    confidence: 0.92,
    subtype: 'Base Set'
  };

  const mockOCRResult: OCRResult = {
    extractedText: "Pikachu\nElectric Mouse Pokémon\nHP 60\nThunder Shock 10\nThunderbolt 30",
    confidence: 0.87,
    textRegions: [
      { text: "Pikachu", confidence: 0.95, boundingBox: { x: 20, y: 15, width: 80, height: 12 } },
      { text: "Electric Mouse Pokémon", confidence: 0.89, boundingBox: { x: 15, y: 30, width: 120, height: 8 } },
      { text: "HP 60", confidence: 0.92, boundingBox: { x: 140, y: 15, width: 30, height: 10 } },
    ]
  };

  const mockIdentificationResult: CardIdentificationResult = {
    cardName: "Pikachu",
    setName: "Base Set",
    setCode: "BS",
    rarity: "Common",
    cardNumber: "58/102",
    confidence: 0.89,
    estimatedValue: {
      low: 5.00,
      market: 12.50,
      high: 25.00,
      currency: "USD"
    }
  };

  const handleRunModel = async (modelType: string) => {
    setIsProcessing(prev => ({ ...prev, [modelType]: true }));
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setIsProcessing(prev => ({ ...prev, [modelType]: false }));
    
    // In a real implementation, this would call the actual model endpoint
    if (onModelResultUpdate) {
      let mockResult;
      switch (modelType) {
        case 'cardType':
          mockResult = mockCardTypeResult;
          break;
        case 'ocr':
          mockResult = mockOCRResult;
          break;
        case 'identification':
          mockResult = mockIdentificationResult;
          break;
      }
      onModelResultUpdate(cardId, modelType, mockResult);
    }
  };

  const renderCardTypeSection = () => (
    <div className="model-section">
      <div className="section-header">
        <h4>Card Type Detection</h4>
        <button 
          className="run-model-btn"
          onClick={() => handleRunModel('cardType')}
          disabled={isProcessing.cardType}
        >
          {isProcessing.cardType ? 'Processing...' : 'Run Detection'}
        </button>
      </div>
      
      <div className="model-content">
        <div className="model-description">
          <p>Identifies the type of trading card game (Pokémon, Magic: The Gathering, Yu-Gi-Oh!, etc.)</p>
        </div>
        
        <div className="prediction-results">
          <div className="prediction-item">
            <div className="prediction-header">
              <span className="card-type pokemon">Pokémon TCG</span>
              <span className="confidence-score">92%</span>
            </div>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: '92%' }}></div>
            </div>
          </div>
          
          <div className="prediction-item">
            <div className="prediction-header">
              <span className="card-type magic">Magic: The Gathering</span>
              <span className="confidence-score">5%</span>
            </div>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: '5%' }}></div>
            </div>
          </div>
          
          <div className="prediction-item">
            <div className="prediction-header">
              <span className="card-type yugioh">Yu-Gi-Oh!</span>
              <span className="confidence-score">2%</span>
            </div>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: '2%' }}></div>
            </div>
          </div>
          
          <div className="prediction-item">
            <div className="prediction-header">
              <span className="card-type sports">Sports Cards</span>
              <span className="confidence-score">1%</span>
            </div>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: '1%' }}></div>
            </div>
          </div>
        </div>
        
        <div className="additional-info">
          <div className="info-item">
            <label>Detected Subtype:</label>
            <span>Base Set Era</span>
          </div>
          <div className="info-item">
            <label>Language:</label>
            <span>English</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderOCRSection = () => (
    <div className="model-section">
      <div className="section-header">
        <h4>Optical Character Recognition (OCR)</h4>
        <button 
          className="run-model-btn"
          onClick={() => handleRunModel('ocr')}
          disabled={isProcessing.ocr}
        >
          {isProcessing.ocr ? 'Processing...' : 'Extract Text'}
        </button>
      </div>
      
      <div className="model-content">
        <div className="model-description">
          <p>Extracts text content from the card including name, abilities, and flavor text</p>
        </div>
        
        <div className="ocr-results">
          <div className="extracted-text">
            <h5>Extracted Text (87% confidence)</h5>
            <div className="text-content">
              <div className="text-line">
                <span className="text-value">Pikachu</span>
                <span className="text-confidence">95%</span>
              </div>
              <div className="text-line">
                <span className="text-value">Electric Mouse Pokémon</span>
                <span className="text-confidence">89%</span>
              </div>
              <div className="text-line">
                <span className="text-value">HP 60</span>
                <span className="text-confidence">92%</span>
              </div>
              <div className="text-line">
                <span className="text-value">Thunder Shock 10</span>
                <span className="text-confidence">84%</span>
              </div>
              <div className="text-line">
                <span className="text-value">Thunderbolt 30</span>
                <span className="text-confidence">88%</span>
              </div>
            </div>
          </div>
          
          <div className="text-regions">
            <h5>Text Regions</h5>
            <div className="regions-list">
              <div className="region-item">
                <span className="region-label">Card Name:</span>
                <span className="region-coords">(20, 15, 80×12)</span>
              </div>
              <div className="region-item">
                <span className="region-label">Description:</span>
                <span className="region-coords">(15, 30, 120×8)</span>
              </div>
              <div className="region-item">
                <span className="region-label">HP:</span>
                <span className="region-coords">(140, 15, 30×10)</span>
              </div>
            </div>
          </div>
          
          <div className="editable-text">
            <h5>Edit Extracted Text</h5>
            <textarea 
              className="text-editor"
              defaultValue="Pikachu&#10;Electric Mouse Pokémon&#10;HP 60&#10;Thunder Shock 10&#10;Thunderbolt 30"
              rows={5}
              placeholder="Edit the extracted text here..."
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderIdentificationSection = () => (
    <div className="model-section">
      <div className="section-header">
        <h4>Card Identification</h4>
        <button 
          className="run-model-btn"
          onClick={() => handleRunModel('identification')}
          disabled={isProcessing.identification}
        >
          {isProcessing.identification ? 'Processing...' : 'Identify Card'}
        </button>
      </div>
      
      <div className="model-content">
        <div className="model-description">
          <p>Identifies the specific card, set, rarity, and provides market value estimates</p>
        </div>
        
        <div className="identification-results">
          <div className="card-identity">
            <h5>Card Identity (89% confidence)</h5>
            <div className="identity-grid">
              <div className="identity-item">
                <label>Card Name:</label>
                <span>Pikachu</span>
              </div>
              <div className="identity-item">
                <label>Set:</label>
                <span>Base Set</span>
              </div>
              <div className="identity-item">
                <label>Set Code:</label>
                <span>BS</span>
              </div>
              <div className="identity-item">
                <label>Card Number:</label>
                <span>58/102</span>
              </div>
              <div className="identity-item">
                <label>Rarity:</label>
                <span className="rarity common">Common</span>
              </div>
              <div className="identity-item">
                <label>Release Year:</label>
                <span>1998</span>
              </div>
            </div>
          </div>
          
          <div className="market-value">
            <h5>Estimated Market Value</h5>
            <div className="value-grid">
              <div className="value-item">
                <label>Low:</label>
                <span className="price">$5.00</span>
              </div>
              <div className="value-item">
                <label>Market:</label>
                <span className="price market-price">$12.50</span>
              </div>
              <div className="value-item">
                <label>High:</label>
                <span className="price">$25.00</span>
              </div>
            </div>
            <div className="value-disclaimer">
              <small>Prices based on recent sales data. Actual value may vary based on condition.</small>
            </div>
          </div>
          
          <div className="similar-cards">
            <h5>Similar Cards</h5>
            <div className="similar-list">
              <div className="similar-item">
                <span className="similar-name">Pikachu (Base Set 2)</span>
                <span className="similar-confidence">76%</span>
              </div>
              <div className="similar-item">
                <span className="similar-name">Pikachu (Jungle)</span>
                <span className="similar-confidence">68%</span>
              </div>
              <div className="similar-item">
                <span className="similar-name">Pikachu (Fossil)</span>
                <span className="similar-confidence">62%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="model-placeholders">
      <div className="placeholders-header">
        <h3>AI Model Analysis</h3>
        <div className="tab-navigation">
          <button 
            className={`tab-btn ${activeTab === 'cardType' ? 'active' : ''}`}
            onClick={() => setActiveTab('cardType')}
          >
            Card Type
          </button>
          <button 
            className={`tab-btn ${activeTab === 'ocr' ? 'active' : ''}`}
            onClick={() => setActiveTab('ocr')}
          >
            OCR
          </button>
          <button 
            className={`tab-btn ${activeTab === 'identification' ? 'active' : ''}`}
            onClick={() => setActiveTab('identification')}
          >
            Identification
          </button>
        </div>
      </div>

      <div className="placeholders-content">
        {activeTab === 'cardType' && renderCardTypeSection()}
        {activeTab === 'ocr' && renderOCRSection()}
        {activeTab === 'identification' && renderIdentificationSection()}
      </div>
    </div>
  );
};
