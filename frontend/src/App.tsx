import React from 'react';
import { CameraInterface } from './components/camera/CameraInterface';
import './App.css';

function App() {
  const handleCapture = (imageData: ImageData) => {
    console.log('Card captured:', imageData);
    // TODO: Process the captured image with ML model
  };

  const handleError = (error: string) => {
    console.error('Camera error:', error);
    // TODO: Show user-friendly error message
  };

  return (
    <div className="App">
      <CameraInterface 
        onCapture={handleCapture}
        onError={handleError}
      />
    </div>
  );
}

export default App;
