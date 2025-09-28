// Debug coordinate conversion
const imageWidth = 1920;
const imageHeight = 1080;
const inputSize = 1088;

// Letterbox parameters (from debug output)
const scale = Math.min(inputSize / imageWidth, inputSize / imageHeight);
const newWidth = Math.round(imageWidth * scale);
const newHeight = Math.round(imageHeight * scale);
const padX = Math.floor((inputSize - newWidth) / 2);
const padY = Math.floor((inputSize - newHeight) / 2);

console.log('📐 Letterbox conversion:');
console.log(`Scale: ${scale.toFixed(3)}`);
console.log(`New size: ${newWidth}x${newHeight}`);
console.log(`Padding: ${padX}, ${padY}`);

// Top detection from debug: cx=345.0, cy=320.9
const modelCx = 345.0;
const modelCy = 320.9;

// Convert to original image coordinates
const imgCx = (modelCx - padX) / scale;
const imgCy = (modelCy - padY) / scale;

console.log('\n🎯 Top detection coordinate conversion:');
console.log(`Model coordinates: (${modelCx}, ${modelCy})`);
console.log(`Image coordinates: (${imgCx.toFixed(1)}, ${imgCy.toFixed(1)})`);
console.log(`Normalized: (${(imgCx/imageWidth).toFixed(3)}, ${(imgCy/imageHeight).toFixed(3)})`);

// The Pokemon card in the image appears to be roughly in the center-right
// Let's see if this makes sense
console.log('\n📊 Expected vs Actual:');
console.log('Expected: Card should be around center-right of image');
console.log(`Actual: Detection at ${(imgCx/imageWidth*100).toFixed(1)}% from left, ${(imgCy/imageHeight*100).toFixed(1)}% from top`);
