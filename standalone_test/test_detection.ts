import * as fs from 'fs';
import * as path from 'path';
import { NodeBackboneModelManager, BackboneModelConfig } from './NodeBackboneModelManager';
import { createCanvas, loadImage } from 'canvas';

// Configuration matching frontend
const BACKBONE_MODEL_CONFIG: BackboneModelConfig = {
  modelPath: '../frontend/public/models/trading_card_detector_backbone.onnx',
  inputSize: 1088, // Model input size from model_info.json
  executionProviders: ['cpu'], // Use CPU for Node.js
  nmsConfig: {
    confidenceThreshold: 0.8,  // High threshold to match validation results (0.82+)
    nmsThreshold: 0.4,         // Stricter NMS threshold
    maxDetections: 3,          // Very few detections for clean results
    inputSize: 1088            // Must match model input size
  }
};

async function runDetectionTest() {
  console.log('🚀 Starting standalone detection test...');
  console.log('📋 Configuration:', BACKBONE_MODEL_CONFIG);

  try {
    // Initialize model manager
    const modelManager = new NodeBackboneModelManager(BACKBONE_MODEL_CONFIG);
    
    // Load model
    console.log('\n📥 Loading ONNX model...');
    await modelManager.loadModel();
    
    // Test image path (use the captured image from the application)
    const testImagePath = '../cam.png';
    
    if (!fs.existsSync(testImagePath)) {
      console.error(`❌ Test image not found: ${testImagePath}`);
      console.log('Available test images:');
      const testImagesDir = '../data/yolo_obb/images/test';
      if (fs.existsSync(testImagesDir)) {
        const files = fs.readdirSync(testImagesDir).slice(0, 10); // Show first 10
        files.forEach(file => console.log(`  - ${file}`));
      }
      return;
    }

    console.log(`\n🖼️ Running inference on: ${testImagePath}`);
    
    // Run detection
    const startTime = Date.now();
    const prediction = await modelManager.predictFromImagePath(testImagePath);
    const totalTime = Date.now() - startTime;

    console.log('\n📊 Detection Results:');
    console.log(`⏱️ Total time: ${totalTime}ms`);
    console.log(`🔥 Inference time: ${prediction.inferenceTime}ms`);
    console.log(`⚙️ Processing time: ${prediction.processingTime}ms`);
    console.log(`🎯 Detections found: ${prediction.detections.length}`);

    // Display detection details
    prediction.detections.forEach((detection, index) => {
      console.log(`\n🔍 Detection ${index + 1}:`);
      console.log(`  📊 Confidence: ${(detection.confidence * 100).toFixed(1)}%`);
      console.log(`  📐 Angle: ${(detection.angle * 180 / Math.PI).toFixed(1)}°`);
      console.log(`  📦 Bounding Box: x=${detection.boundingBox.x.toFixed(3)}, y=${detection.boundingBox.y.toFixed(3)}, w=${detection.boundingBox.width.toFixed(3)}, h=${detection.boundingBox.height.toFixed(3)}`);
      console.log(`  🔄 Corners: (${detection.corners.x1.toFixed(3)},${detection.corners.y1.toFixed(3)}) → (${detection.corners.x2.toFixed(3)},${detection.corners.y2.toFixed(3)}) → (${detection.corners.x3.toFixed(3)},${detection.corners.y3.toFixed(3)}) → (${detection.corners.x4.toFixed(3)},${detection.corners.y4.toFixed(3)})`);
    });

    // Load Python validation results for comparison
    const validationResultsPath = '../validation_results.json';
    if (fs.existsSync(validationResultsPath)) {
      console.log('\n🔍 Comparing with Python validation results...');
      const validationData = JSON.parse(fs.readFileSync(validationResultsPath, 'utf8'));
      
      if (validationData.detections && validationData.detections.length > 0) {
        console.log(`📊 Python detections: ${validationData.detections.length}`);
        console.log(`📊 TypeScript detections: ${prediction.detections.length}`);
        
        // Compare first detection if available
        if (prediction.detections.length > 0 && validationData.detections.length > 0) {
          const tsDetection = prediction.detections[0];
          const pyDetection = validationData.detections[0];
          
          console.log('\n🔍 Comparing first detection:');
          console.log(`  📊 Confidence - TS: ${(tsDetection.confidence * 100).toFixed(1)}%, Python: ${(pyDetection.confidence * 100).toFixed(1)}%`);
          console.log(`  📐 Angle - TS: ${(tsDetection.angle * 180 / Math.PI).toFixed(1)}°, Python: ${(pyDetection.angle * 180 / Math.PI).toFixed(1)}°`);
          
          // Calculate IoU between bounding boxes
          const iou = calculateIoU(tsDetection.boundingBox, pyDetection.bbox);
          console.log(`  📦 IoU: ${(iou * 100).toFixed(1)}%`);
          
          if (iou > 0.8) {
            console.log('  ✅ Detection alignment: EXCELLENT');
          } else if (iou > 0.5) {
            console.log('  ⚠️ Detection alignment: GOOD');
          } else {
            console.log('  ❌ Detection alignment: POOR');
          }
        }
      }
    } else {
      console.log('⚠️ Python validation results not found - run validation script first');
    }

    // Generate visual output
    await generateVisualOutput(testImagePath, prediction.detections);

    // Clean up
    await modelManager.dispose();
    console.log('\n✅ Test completed successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

function calculateIoU(
  box1: { x: number; y: number; width: number; height: number },
  box2: { x: number; y: number; width: number; height: number }
): number {
  // Calculate intersection
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

  if (x2 < x1 || y2 < y1) return 0;

  const intersectionArea = (x2 - x1) * (y2 - y1);
  const box1Area = box1.width * box1.height;
  const box2Area = box2.width * box2.height;
  const unionArea = box1Area + box2Area - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

async function generateVisualOutput(imagePath: string, detections: any[]) {
  try {
    console.log('\n🎨 Generating visual output...');
    
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    
    // Draw original image
    ctx.drawImage(image, 0, 0);
    
    // Draw detections
    detections.forEach((detection, index) => {
      const { corners, confidence } = detection;
      
      // Convert normalized coordinates to pixel coordinates
      const x1 = corners.x1 * image.width;
      const y1 = corners.y1 * image.height;
      const x2 = corners.x2 * image.width;
      const y2 = corners.y2 * image.height;
      const x3 = corners.x3 * image.width;
      const y3 = corners.y3 * image.height;
      const x4 = corners.x4 * image.width;
      const y4 = corners.y4 * image.height;
      
      // Draw oriented bounding box
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.lineTo(x4, y4);
      ctx.closePath();
      ctx.stroke();
      
      // Draw confidence label
      ctx.fillStyle = '#00ff00';
      ctx.font = '20px Arial';
      ctx.fillText(`${(confidence * 100).toFixed(1)}%`, x1, y1 - 10);
    });
    
    // Save output
    const outputPath = './standalone_test_output.png';
    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(outputPath, buffer);
    
    console.log(`✅ Visual output saved to: ${outputPath}`);
    
  } catch (error) {
    console.error('❌ Failed to generate visual output:', error);
  }
}

// Run the test
if (require.main === module) {
  runDetectionTest().catch(console.error);
}
