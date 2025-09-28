import * as fs from 'fs';
import { NodeBackboneModelManager, BackboneModelConfig } from './NodeBackboneModelManager';
import { createCanvas, loadImage } from 'canvas';

const BACKBONE_MODEL_CONFIG: BackboneModelConfig = {
  modelPath: '../frontend/public/models/trading_card_detector_backbone.onnx',
  inputSize: 1088,
  executionProviders: ['cpu'],
  nmsConfig: {
    confidenceThreshold: 0.8,
    nmsThreshold: 0.4,
    maxDetections: 3,
    inputSize: 1088
  }
};

async function debugJavaScriptDetailed() {
  console.log('🔍 DETAILED JAVASCRIPT DEBUGGING');
  
  try {
    const modelManager = new NodeBackboneModelManager(BACKBONE_MODEL_CONFIG);
    await modelManager.loadModel();
    
    const testImagePath = '../cam.png';
    const image = await loadImage(testImagePath);
    console.log(`Original image: ${image.width}x${image.height}`);
    
    // Get access to internal methods for step-by-step debugging
    const session = (modelManager as any).session;
    
    // Step 1: Load and convert image to ImageData
    console.log('\n📸 STEP 1: Image Loading');
    const imageData = await (modelManager as any).imageToImageData(image);
    console.log(`ImageData: ${imageData.width}x${imageData.height}, data length: ${imageData.data.length}`);
    
    // Sample some pixels from ImageData
    const centerPixelIdx = (544 * imageData.width + 544) * 4;
    console.log(`Center pixel RGBA: [${imageData.data[centerPixelIdx]}, ${imageData.data[centerPixelIdx+1]}, ${imageData.data[centerPixelIdx+2]}, ${imageData.data[centerPixelIdx+3]}]`);
    
    // Step 2: Preprocess image to tensor
    console.log('\n🔄 STEP 2: Preprocessing');
    const inputTensor = (modelManager as any).preprocessImage(imageData);
    
    console.log(`Input tensor shape: ${inputTensor.dims}`);
    console.log(`Input tensor data length: ${inputTensor.data.length}`);
    
    const tensorData = inputTensor.data as Float32Array;
    let min = tensorData[0], max = tensorData[0];
    for (let i = 0; i < tensorData.length; i++) {
      if (tensorData[i] < min) min = tensorData[i];
      if (tensorData[i] > max) max = tensorData[i];
    }
    console.log(`Input tensor range: [${min.toFixed(6)}, ${max.toFixed(6)}]`);
    
    // Sample center pixel from tensor (should match ImageData after normalization)
    const centerTensorIdx = 544 * 1088 + 544;
    const tensorR = tensorData[centerTensorIdx];
    const tensorG = tensorData[centerTensorIdx + 1088 * 1088];
    const tensorB = tensorData[centerTensorIdx + 2 * 1088 * 1088];
    console.log(`Center pixel tensor RGB: [${tensorR.toFixed(6)}, ${tensorG.toFixed(6)}, ${tensorB.toFixed(6)}]`);
    
    // Verify conversion: ImageData RGBA -> Tensor RGB
    const expectedR = imageData.data[centerPixelIdx] / 255.0;
    const expectedG = imageData.data[centerPixelIdx + 1] / 255.0;
    const expectedB = imageData.data[centerPixelIdx + 2] / 255.0;
    console.log(`Expected RGB: [${expectedR.toFixed(6)}, ${expectedG.toFixed(6)}, ${expectedB.toFixed(6)}]`);
    console.log(`Conversion match: R=${Math.abs(tensorR - expectedR) < 0.001}, G=${Math.abs(tensorG - expectedG) < 0.001}, B=${Math.abs(tensorB - expectedB) < 0.001}`);
    
    // Save input tensor for comparison with Python
    const inputBuffer = Buffer.from(tensorData.buffer);
    fs.writeFileSync('../javascript_input_tensor_debug.bin', inputBuffer);
    console.log('Saved: javascript_input_tensor_debug.bin');
    
    // Step 3: Run inference
    console.log('\n🧠 STEP 3: Model Inference');
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
    
    // Analyze output channels
    const [batch, channels, anchors] = output.dims as number[];
    console.log(`Parsed shape: batch=${batch}, channels=${channels}, anchors=${anchors}`);
    
    const channelNames = ['cx', 'cy', 'w', 'h', 'angle', 'conf'];
    for (let ch = 0; ch < channels; ch++) {
      const channelStart = ch * anchors;
      const channelData = outputData.slice(channelStart, channelStart + anchors);
      
      let chMin = channelData[0], chMax = channelData[0], chSum = 0;
      for (let i = 0; i < channelData.length; i++) {
        if (channelData[i] < chMin) chMin = channelData[i];
        if (channelData[i] > chMax) chMax = channelData[i];
        chSum += channelData[i];
      }
      const chMean = chSum / channelData.length;
      
      console.log(`Channel ${ch} (${channelNames[ch]}): min=${chMin.toFixed(3)}, max=${chMax.toFixed(3)}, mean=${chMean.toFixed(3)}`);
    }
    
    // Save output tensor for comparison with Python
    const outputBuffer = Buffer.from(outputData.buffer);
    fs.writeFileSync('../javascript_output_tensor_debug.bin', outputBuffer);
    console.log('Saved: javascript_output_tensor_debug.bin');
    
    // Step 4: Find high-confidence detections (before NMS)
    console.log('\n🎯 STEP 4: High-Confidence Detection Analysis');
    const confChannel = outputData.slice(5 * anchors, 6 * anchors);
    const sigmoidConf = Array.from(confChannel).map(v => 1 / (1 + Math.exp(-v)));
    const highConfIndices = sigmoidConf
      .map((conf, idx) => ({ conf, idx }))
      .filter(item => item.conf > 0.8)
      .sort((a, b) => b.conf - a.conf)
      .slice(0, 10);
    
    console.log(`High confidence detections (>0.8): ${highConfIndices.length}`);
    
    const detections = [];
    for (let i = 0; i < Math.min(3, highConfIndices.length); i++) {
      const { conf, idx } = highConfIndices[i];
      const cx = outputData[0 * anchors + idx];
      const cy = outputData[1 * anchors + idx];
      const w = outputData[2 * anchors + idx];
      const h = outputData[3 * anchors + idx];
      const angle = outputData[4 * anchors + idx];
      
      // Convert to image coordinates (same as Python)
      const scale = 0.5666666666666667;
      const padX = 0, padY = 238;
      const imgCx = (cx - padX) / scale;
      const imgCy = (cy - padY) / scale;
      
      const detection = {
        rank: i + 1,
        anchor_idx: idx,
        model_coords: { cx, cy, w, h, angle },
        confidence: conf,
        image_coords: { cx: imgCx, cy: imgCy },
        image_percent: { cx: imgCx/1920*100, cy: imgCy/1080*100 }
      };
      detections.push(detection);
      
      console.log(`Detection ${i+1}: anchor=${idx}, conf=${conf.toFixed(3)}, model=(${cx.toFixed(1)},${cy.toFixed(1)}), image=(${(imgCx/1920*100).toFixed(1)}%,${(imgCy/1080*100).toFixed(1)}%)`);
    }
    
    // Save JavaScript results for comparison
    const results_data = {
      image_info: { width: 1920, height: 1080, path: '../cam.png' },
      preprocessing: { input_size: 1088, scale: 0.5666666666666667, padding: [0, 238] },
      model_output: { shape: [batch, channels, anchors], total_anchors: anchors },
      detections: detections
    };
    
    fs.writeFileSync('../javascript_results_debug.json', JSON.stringify(results_data, null, 2));
    console.log('Saved: javascript_results_debug.json');
    
    await modelManager.dispose();
    
  } catch (error) {
    console.error('❌ Debug failed:', error);
  }
}

if (require.main === module) {
  debugJavaScriptDetailed().catch(console.error);
}
