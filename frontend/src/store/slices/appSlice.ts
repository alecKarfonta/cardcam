import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export type AppView = 'camera' | 'extraction' | 'gallery' | 'settings' | 'help';

export interface AppNotification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: number;
  autoHide?: boolean;
  duration?: number;
}

interface AppSliceState {
  currentView: AppView;
  isOnline: boolean;
  notifications: AppNotification[];
  isFullscreen: boolean;
  theme: 'light' | 'dark' | 'auto';
  debugMode: boolean;
  performanceMetrics: {
    fps: number;
    memoryUsage: number;
    batteryLevel: number;
  };
}

const initialState: AppSliceState = {
  currentView: 'camera',
  isOnline: navigator.onLine,
  notifications: [],
  isFullscreen: false,
  theme: 'dark',
  debugMode: process.env.NODE_ENV === 'development',
  performanceMetrics: {
    fps: 0,
    memoryUsage: 0,
    batteryLevel: 1,
  },
};

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setCurrentView: (state, action: PayloadAction<AppView>) => {
      state.currentView = action.payload;
    },
    
    setOnlineStatus: (state, action: PayloadAction<boolean>) => {
      state.isOnline = action.payload;
    },
    
    addNotification: (state, action: PayloadAction<Omit<AppNotification, 'id' | 'timestamp'>>) => {
      const notification: AppNotification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: Date.now(),
      };
      state.notifications.push(notification);
    },
    
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    
    clearNotifications: (state) => {
      state.notifications = [];
    },
    
    setFullscreen: (state, action: PayloadAction<boolean>) => {
      state.isFullscreen = action.payload;
    },
    
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'auto'>) => {
      state.theme = action.payload;
    },
    
    setDebugMode: (state, action: PayloadAction<boolean>) => {
      state.debugMode = action.payload;
    },
    
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<AppSliceState['performanceMetrics']>>) => {
      state.performanceMetrics = {
        ...state.performanceMetrics,
        ...action.payload,
      };
    },
  },
});

export const {
  setCurrentView,
  setOnlineStatus,
  addNotification,
  removeNotification,
  clearNotifications,
  setFullscreen,
  setTheme,
  setDebugMode,
  updatePerformanceMetrics,
} = appSlice.actions;

export default appSlice.reducer;
