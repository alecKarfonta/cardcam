import { configureStore } from '@reduxjs/toolkit';
import cameraReducer from './slices/cameraSlice';
import inferenceReducer from './slices/inferenceSlice';
import appReducer from './slices/appSlice';

export const store = configureStore({
  reducer: {
    camera: cameraReducer,
    inference: inferenceReducer,
    app: appReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['camera/setStream'],
        // Ignore these field paths in all actions
        ignoredActionsPaths: ['payload.stream'],
        // Ignore these paths in the state
        ignoredPaths: ['camera.stream'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
