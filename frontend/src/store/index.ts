import { configureStore } from '@reduxjs/toolkit';
import cameraReducer from './slices/cameraSlice';
import inferenceReducer from './slices/inferenceSlice';
import appReducer from './slices/appSlice';
import cardExtractionReducer from './slices/cardExtractionSlice';

export const store = configureStore({
  reducer: {
    camera: cameraReducer,
    inference: inferenceReducer,
    app: appReducer,
    cardExtraction: cardExtractionReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['camera/setStream', 'cardExtraction/addExtractedCard'],
        // Ignore these field paths in all actions
        ignoredActionsPaths: ['payload.stream', 'payload.imageData'],
        // Ignore these paths in the state
        ignoredPaths: ['camera.stream', 'cardExtraction.currentSession.extractedCards', 'cardExtraction.extractionHistory'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
