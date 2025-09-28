import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { CardDetection } from './inferenceSlice';

export interface ExtractedCard {
  id: string;
  imageData: ImageData;
  originalDetection: CardDetection;
  extractedAt: number;
  dimensions: {
    width: number;
    height: number;
  };
  metadata?: {
    cardName?: string;
    setName?: string;
    rarity?: string;
    condition?: string;
  };
  // Enhanced metadata from high-resolution extraction
  extractionMetadata?: {
    cardId: string;
    extractionMethod: 'bbox' | 'obb' | 'perspective' | 'fallback';
    originalSize: { width: number; height: number };
    modelInputSize: { width: number; height: number };
    scalingFactors: { x: number; y: number };
    confidence: number;
    rotationAngle?: number;
    corners?: Array<{ x: number; y: number }>;
    paddingApplied: { x: number; y: number };
    isHighResolution: boolean;
  };
  qualityScore?: number;
  qualityFactors?: string[];
}

export interface ExtractionSession {
  id: string;
  timestamp: number;
  sourceImageDimensions: {
    width: number;
    height: number;
  };
  extractedCards: ExtractedCard[];
  totalDetections: number;
  successfulExtractions: number;
}

interface CardExtractionSliceState {
  currentSession: ExtractionSession | null;
  extractionHistory: ExtractionSession[];
  isExtracting: boolean;
  extractionProgress: {
    current: number;
    total: number;
    status: string;
  };
  maxHistorySize: number;
  selectedCardId: string | null;
}

const initialState: CardExtractionSliceState = {
  currentSession: null,
  extractionHistory: [],
  isExtracting: false,
  extractionProgress: {
    current: 0,
    total: 0,
    status: 'idle'
  },
  maxHistorySize: 10,
  selectedCardId: null,
};

const cardExtractionSlice = createSlice({
  name: 'cardExtraction',
  initialState,
  reducers: {
    startExtraction: (state, action: PayloadAction<{
      sourceImageDimensions: { width: number; height: number };
      totalDetections: number;
    }>) => {
      const sessionId = `session_${Date.now()}`;
      
      state.currentSession = {
        id: sessionId,
        timestamp: Date.now(),
        sourceImageDimensions: action.payload.sourceImageDimensions,
        extractedCards: [],
        totalDetections: action.payload.totalDetections,
        successfulExtractions: 0,
      };
      
      state.isExtracting = true;
      state.extractionProgress = {
        current: 0,
        total: action.payload.totalDetections,
        status: 'extracting'
      };
    },

    updateExtractionProgress: (state, action: PayloadAction<{
      current: number;
      status: string;
    }>) => {
      state.extractionProgress.current = action.payload.current;
      state.extractionProgress.status = action.payload.status;
    },

    addExtractedCard: (state, action: PayloadAction<ExtractedCard>) => {
      if (state.currentSession) {
        state.currentSession.extractedCards.push(action.payload);
        state.currentSession.successfulExtractions += 1;
      }
    },

    completeExtraction: (state) => {
      state.isExtracting = false;
      state.extractionProgress.status = 'completed';
      
      if (state.currentSession) {
        // Add to history
        state.extractionHistory.unshift(state.currentSession);
        
        // Limit history size
        if (state.extractionHistory.length > state.maxHistorySize) {
          state.extractionHistory = state.extractionHistory.slice(0, state.maxHistorySize);
        }
      }
    },

    cancelExtraction: (state) => {
      state.isExtracting = false;
      state.extractionProgress = {
        current: 0,
        total: 0,
        status: 'cancelled'
      };
      state.currentSession = null;
    },

    selectCard: (state, action: PayloadAction<string | null>) => {
      state.selectedCardId = action.payload;
    },

    updateCardMetadata: (state, action: PayloadAction<{
      cardId: string;
      metadata: ExtractedCard['metadata'];
    }>) => {
      if (state.currentSession) {
        const card = state.currentSession.extractedCards.find(c => c.id === action.payload.cardId);
        if (card) {
          card.metadata = {
            ...card.metadata,
            ...action.payload.metadata
          };
        }
      }
      
      // Also update in history
      for (const session of state.extractionHistory) {
        const card = session.extractedCards.find(c => c.id === action.payload.cardId);
        if (card) {
          card.metadata = {
            ...card.metadata,
            ...action.payload.metadata
          };
          break;
        }
      }
    },

    deleteExtractedCard: (state, action: PayloadAction<string>) => {
      const cardId = action.payload;
      
      if (state.currentSession) {
        state.currentSession.extractedCards = state.currentSession.extractedCards.filter(
          c => c.id !== cardId
        );
        state.currentSession.successfulExtractions = state.currentSession.extractedCards.length;
      }
      
      // Also remove from history
      for (const session of state.extractionHistory) {
        session.extractedCards = session.extractedCards.filter(c => c.id !== cardId);
        session.successfulExtractions = session.extractedCards.length;
      }
      
      // Clear selection if deleted card was selected
      if (state.selectedCardId === cardId) {
        state.selectedCardId = null;
      }
    },

    clearCurrentSession: (state) => {
      state.currentSession = null;
      state.selectedCardId = null;
      state.extractionProgress = {
        current: 0,
        total: 0,
        status: 'idle'
      };
    },

    clearExtractionHistory: (state) => {
      state.extractionHistory = [];
    },

    loadSessionFromHistory: (state, action: PayloadAction<string>) => {
      const session = state.extractionHistory.find(s => s.id === action.payload);
      if (session) {
        state.currentSession = { ...session };
        state.selectedCardId = null;
      }
    },

    resetExtraction: (state) => {
      return initialState;
    },
  },
});

export const {
  startExtraction,
  updateExtractionProgress,
  addExtractedCard,
  completeExtraction,
  cancelExtraction,
  selectCard,
  updateCardMetadata,
  deleteExtractedCard,
  clearCurrentSession,
  clearExtractionHistory,
  loadSessionFromHistory,
  resetExtraction,
} = cardExtractionSlice.actions;

export default cardExtractionSlice.reducer;
