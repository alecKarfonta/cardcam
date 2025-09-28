import * as fs from 'fs';
import { NodeBackboneModelManager, BackboneModelConfig } from './NodeBackboneModelManager';
import { createCanvas, loadImage } from 'canvas';

const BACKBONE_MODEL_CONFIG: BackboneModelConfig = {
  modelPath: '../frontend/public/models/trading_card_detector_backbone_opset18.onnx',
  inputSize: 1088,
  executionProviders: ['cpu'],
  nmsConfig: {
    confidenceThreshold: 0.1,
    nmsThreshold: 0.4,
    maxDetections: 10,
    inputSize: 1088
  }
};

async function debugTensorSave() {
  console.log('🔍 JAVASCRIPT TENSOR DEBUG');
  
  try {
    const modelManager = new NodeBackboneModelManager(BACKBONE_MODEL_CONFIG);
    await modelManager.loadModel();
    
    const testImagePath = '../cam.png';
    const image = await loadImage(testImagePath);
    console.log(`Original image: ${image.width}x${image.height}`);
    
    // Get access to internal methods
    const session = (modelManager as any).session;
    
    // Load and preprocess image exactly like the manager does
    const imageData = await (modelManager as any).imageToImageData(image);
    const inputTensor = (modelManager as any).preprocessImage(imageData);
    
    console.log(`Input tensor shape: ${inputTensor.dims}`);
    console.log(`Input tensor data length: ${inputTensor.data.length}`);
    console.log(`Input tensor data type: ${inputTensor.type}`);
    
    // Sample some values
    const tensorData = inputTensor.data as Float32Array;
    let min = tensorData[0], max = tensorData[0];
    for (let i = 0; i < tensorData.length; i++) {
      if (tensorData[i] < min) min = tensorData[i];
      if (tensorData[i] > max) max = tensorData[i];
    }
    console.log(`Input tensor range: [${min.toFixed(6)}, ${max.toFixed(6)}]`);
    console.log(`Sample pixels (first 10 from R channel): ${Array.from(tensorData.slice(0, 10))}`);
    console.log(`Sample pixels (center area): ${Array.from(tensorData.slice(544 * 1088 + 540, 544 * 1088 + 550))}`);
    
    // Save input tensor
    const inputBuffer = Buffer.from(tensorData.buffer);
    fs.writeFileSync('../javascript_input_tensor.bin', inputBuffer);
    console.log('Saved input tensor to javascript_input_tensor.bin');
    
    // Run inference
    const results = await session.run({
      [session.inputNames[0]]: inputTensor
    });
    
    const output = results[Object.keys(results)[0]];
    const outputData = output.data as Float32Array;
    
    console.log(`Output shape: ${output.dims}`);
    let outMin = outputData[0], outMax = outputData[0];
    for (let i = 0; i < outputData.length; i++) {
      if (outputData[i] < outMin) outMin = outputData[i];
      if (outputData[i] > outMax) outMax = outputData[i];
    }
    console.log(`Output range: [${outMin.toFixed(6)}, ${outMax.toFixed(6)}]`);
    
    // Save output tensor
    const outputBuffer = Buffer.from(outputData.buffer);
    fs.writeFileSync('../javascript_output_tensor.bin', outputBuffer);
    console.log('Saved output tensor to javascript_output_tensor.bin');
    
    // Check specific values
    console.log(`First 10 cx values: ${Array.from(outputData.slice(0, 10))}`);
    console.log(`First 10 cy values: ${Array.from(outputData.slice(24276, 24286))}`);
    console.log(`First 10 confidence values: ${Array.from(outputData.slice(24276 * 5, 24276 * 5 + 10))}`);
    
    // Check the high-confidence detection that JavaScript found
    const js_high_conf_idx = 5483; // From our earlier debug
    if (js_high_conf_idx < 24276) {
      const cx = outputData[js_high_conf_idx];
      const cy = outputData[24276 + js_high_conf_idx];
      const w = outputData[24276 * 2 + js_high_conf_idx];
      const h = outputData[24276 * 3 + js_high_conf_idx];
      const angle = outputData[24276 * 4 + js_high_conf_idx];
      const conf_raw = outputData[24276 * 5 + js_high_conf_idx];
      const conf = 1 / (1 + Math.exp(-conf_raw));
      
      console.log(`JavaScript high confidence detection at idx ${js_high_conf_idx}:`);
      console.log(`  cx=${cx.toFixed(3)}, cy=${cy.toFixed(3)}, w=${w.toFixed(3)}, h=${h.toFixed(3)}, angle=${angle.toFixed(6)}, conf=${conf.toFixed(3)}`);
    }
    
    await modelManager.dispose();
    
  } catch (error) {
    console.error('❌ Debug failed:', error);
  }
}

if (require.main === module) {
  debugTensorSave().catch(console.error);
}
