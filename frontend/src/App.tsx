import React, { useRef } from 'react';
import { useAppSelector, useAppDispatch } from './hooks/redux';
import { CameraInterface, CameraInterfaceRef } from './components/camera/CameraInterface';
import { CardExtractionView } from './components/extraction/CardExtractionView';
import { setCurrentView } from './store/slices/appSlice';
import { RootState } from './store';
import './App.css';

function App() {
  const dispatch = useAppDispatch();
  const currentView = useAppSelector((state: RootState) => state.app.currentView);
  const extractionState = useAppSelector((state: RootState) => state.cardExtraction);
  const cameraRef = useRef<CameraInterfaceRef>(null);

  const handleCapture = (imageData: ImageData) => {
    console.log('Card captured:', imageData);
    // Card extraction is now handled in CameraInterface
  };

  const handleError = (error: string) => {
    console.error('Camera error:', error);
    // TODO: Show user-friendly error message
  };

  const handleBackToCamera = () => {
    dispatch(setCurrentView('camera'));
    // Resume the camera when returning from extraction view
    if (cameraRef.current) {
      cameraRef.current.resumeCamera();
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'extraction':
        return (
          <CardExtractionView
            extractedCards={extractionState.currentSession?.extractedCards || []}
            onBack={handleBackToCamera}
          />
        );
      case 'camera':
      default:
        return (
          <CameraInterface 
            ref={cameraRef}
            onCapture={handleCapture}
            onError={handleError}
          />
        );
    }
  };

  return (
    <div className="App">
      {renderCurrentView()}
    </div>
  );
}

export default App;
