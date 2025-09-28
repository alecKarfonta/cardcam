import * as fs from 'fs';
import { NodeBackboneModelManager, BackboneModelConfig } from './NodeBackboneModelManager';

const BACKBONE_MODEL_CONFIG: BackboneModelConfig = {
  modelPath: '../frontend/public/models/trading_card_detector_backbone_opset18.onnx',
  inputSize: 1088,
  executionProviders: ['cpu'],
  nmsConfig: {
    confidenceThreshold: 0.1,  // Lower threshold to see more detections
    nmsThreshold: 0.4,
    maxDetections: 10,  // More detections to analyze
    inputSize: 1088
  }
};

async function debugModelOutput() {
  console.log('🔍 DEBUGGING MODEL OUTPUT FORMAT');
  
  try {
    const modelManager = new NodeBackboneModelManager(BACKBONE_MODEL_CONFIG);
    await modelManager.loadModel();
    
    const testImagePath = '../cam.png';
    console.log(`\n🖼️ Loading image: ${testImagePath}`);
    
    // Get image info
    const { createCanvas, loadImage } = require('canvas');
    const image = await loadImage(testImagePath);
    console.log(`📐 Original image: ${image.width}x${image.height}`);
    
    // Create a custom version that exposes raw model outputs
    const session = (modelManager as any).session;
    
    // Load and preprocess image
    const imageData = (modelManager as any).imageToImageData(image);
    const inputTensor = (modelManager as any).preprocessImage(imageData);
    
    console.log(`📊 Input tensor shape: ${inputTensor.dims}`);
    console.log(`📊 Input tensor data length: ${inputTensor.data.length}`);
    
    // Run inference
    const results = await session.run({
      [session.inputNames[0]]: inputTensor
    });
    
    const outputNames = Object.keys(results);
    const output = results[outputNames[0]];
    const outputData = output.data as Float32Array;
    
    console.log('\n📊 RAW MODEL OUTPUT ANALYSIS:');
    console.log(`📊 Output tensor shape: ${output.dims}`);
    console.log(`📊 Output data length: ${outputData.length}`);
    
    // Analyze the shape
    const [batch, channels, anchors] = output.dims as number[];
    console.log(`📊 Parsed shape: batch=${batch}, channels=${channels}, anchors=${anchors}`);
    
    // Expected: (1, 6, 8400) but we might be getting different format
    if (channels === 6) {
      console.log('✅ Expected 6 channels for OBB: [cx, cy, w, h, angle, confidence]');
    } else {
      console.log(`❌ Unexpected channel count: ${channels}`);
    }
    
    // Analyze each channel
    const chStride = anchors;
    for (let ch = 0; ch < channels; ch++) {
      const channelData = outputData.slice(ch * chStride, (ch + 1) * chStride);
      const min = Math.min.apply(null, Array.from(channelData));
      const max = Math.max.apply(null, Array.from(channelData));
      const mean = Array.from(channelData).reduce((a, b) => a + b) / channelData.length;
      
      const channelNames = ['cx', 'cy', 'w', 'h', 'angle', 'confidence'];
      const channelName = channelNames[ch] || `ch${ch}`;
      
      console.log(`📊 Channel ${ch} (${channelName}): min=${min.toFixed(3)}, max=${max.toFixed(3)}, mean=${mean.toFixed(3)}`);
      
      // For confidence channel, check sigmoid values
      if (ch === 5) {
        const sigmoidValues = Array.from(channelData.slice(0, 10)).map(v => 1 / (1 + Math.exp(-v)));
        console.log(`   Sigmoid(first 10): ${sigmoidValues.map(v => v.toFixed(3)).join(', ')}`);
      }
    }
    
    // Look for high-confidence detections
    const confChannel = outputData.slice(5 * chStride, 6 * chStride);
    const sigmoidConf = Array.from(confChannel).map(v => 1 / (1 + Math.exp(-v)));
    const highConfIndices = sigmoidConf
      .map((conf, idx) => ({ conf, idx }))
      .filter(item => item.conf > 0.5)
      .sort((a, b) => b.conf - a.conf)
      .slice(0, 10);
    
    console.log('\n🎯 TOP 10 HIGH-CONFIDENCE DETECTIONS:');
    highConfIndices.forEach((item, rank) => {
      const { conf, idx } = item;
      const cx = outputData[0 * chStride + idx];
      const cy = outputData[1 * chStride + idx];
      const w = outputData[2 * chStride + idx];
      const h = outputData[3 * chStride + idx];
      const angle = outputData[4 * chStride + idx];
      
      console.log(`${rank + 1}. Anchor ${idx}: conf=${conf.toFixed(3)}, cx=${cx.toFixed(1)}, cy=${cy.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}, angle=${angle.toFixed(1)}`);
    });
    
    await modelManager.dispose();
    
  } catch (error) {
    console.error('❌ Debug failed:', error);
  }
}

if (require.main === module) {
  debugModelOutput().catch(console.error);
}
