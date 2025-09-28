/**
 * High-Resolution Card Extraction Demo
 * 
 * This utility demonstrates the high-resolution card extraction functionality
 * and provides examples of how to use the new extraction system.
 */

import { CardDetection } from '../store/slices/inferenceSlice';
import { HighResCardExtractor } from './HighResCardExtractor';
import { EnhancedCardCropper } from './EnhancedCardCropper';

/**
 * Demo function showing high-resolution extraction capabilities
 */
export async function demonstrateHighResExtraction() {
  console.log('🎯 High-Resolution Card Extraction Demo');
  console.log('=====================================');

  // Example detection results (simulating YOLO OBB output)
  const mockDetections: CardDetection[] = [
    {
      id: 'card_1',
      boundingBox: { x: 0.2, y: 0.3, width: 0.15, height: 0.25 },
      confidence: 0.85,
      corners: [
        { x: 0.2, y: 0.3 },   // Top-left
        { x: 0.35, y: 0.28 }, // Top-right (slightly rotated)
        { x: 0.37, y: 0.55 }, // Bottom-right
        { x: 0.22, y: 0.57 }  // Bottom-left
      ],
      isRotated: true
    },
    {
      id: 'card_2',
      boundingBox: { x: 0.6, y: 0.4, width: 0.2, height: 0.3 },
      confidence: 0.92,
      isRotated: false
    }
  ];

  console.log(`📊 Mock detections created: ${mockDetections.length} cards`);
  console.log('   - Card 1: Rotated with corners (OBB extraction)');
  console.log('   - Card 2: Axis-aligned (bbox extraction)');

  // Simulate model input size vs original image size
  const modelInputSize = { width: 1024, height: 768 };
  const originalImageSize = { width: 1920, height: 1080 };
  
  const scalingFactors = {
    x: originalImageSize.width / modelInputSize.width,
    y: originalImageSize.height / modelInputSize.height
  };

  console.log(`\n📐 Resolution comparison:`);
  console.log(`   Model Input: ${modelInputSize.width}x${modelInputSize.height}`);
  console.log(`   Original: ${originalImageSize.width}x${originalImageSize.height}`);
  console.log(`   Scaling: ${scalingFactors.x.toFixed(2)}x, ${scalingFactors.y.toFixed(2)}x`);

  // Calculate expected improvements
  const card1_modelRes = {
    width: Math.round(0.15 * modelInputSize.width),
    height: Math.round(0.25 * modelInputSize.height)
  };
  
  const card1_highRes = {
    width: Math.round(0.15 * originalImageSize.width),
    height: Math.round(0.25 * originalImageSize.height)
  };

  console.log(`\n🎯 Expected extraction improvements:`);
  console.log(`   Card 1 - Model resolution: ${card1_modelRes.width}x${card1_modelRes.height} (${card1_modelRes.width * card1_modelRes.height} pixels)`);
  console.log(`   Card 1 - High resolution: ${card1_highRes.width}x${card1_highRes.height} (${card1_highRes.width * card1_highRes.height} pixels)`);
  
  const improvementRatio = (card1_highRes.width * card1_highRes.height) / (card1_modelRes.width * card1_modelRes.height);
  console.log(`   Improvement: ${improvementRatio.toFixed(1)}x more pixels!`);

  // Show extraction method selection
  console.log(`\n🔧 Extraction method selection:`);
  mockDetections.forEach((detection, index) => {
    if (detection.isRotated && detection.corners) {
      console.log(`   Card ${index + 1}: Perspective correction (rotated card with corners)`);
    } else {
      console.log(`   Card ${index + 1}: Bounding box extraction (axis-aligned)`);
    }
  });

  // Show quality scoring factors
  console.log(`\n⭐ Quality scoring factors:`);
  console.log(`   - Detection confidence: up to 40 points`);
  console.log(`   - High-resolution bonus: +20 points`);
  console.log(`   - Large size bonus: up to +20 points`);
  console.log(`   - Perspective correction: +15 points`);
  console.log(`   - Good aspect ratio: +5 points`);
  console.log(`   Total possible: 100 points`);

  return {
    mockDetections,
    modelInputSize,
    originalImageSize,
    scalingFactors,
    expectedImprovement: improvementRatio
  };
}

/**
 * Example of how to use the high-resolution extraction in practice
 */
export async function exampleUsage() {
  console.log('\n📝 Example Usage:');
  console.log('================');

  console.log(`
// 1. In camera capture handler:
const cropResults = await EnhancedCardCropper.extractFromCameraFrame(
  processedFrame,           // Frame used for inference (1024x768)
  detections,              // Detection results from model
  videoElement,            // Video element for native resolution
  {
    modelInputSize: { width: 1024, height: 768 },
    paddingRatio: 0.05,    // 5% padding around cards
    enablePerspectiveCorrection: true
  }
);

// 2. Access enhanced metadata:
cropResults.forEach((result, index) => {
  console.log(\`Card \${index + 1}:\`);
  console.log(\`  Size: \${result.extractedWidth}x\${result.extractedHeight}\`);
  console.log(\`  Method: \${result.metadata.extractionMethod}\`);
  console.log(\`  High-res: \${result.metadata.isHighResolution}\`);
  console.log(\`  Quality: \${EnhancedCardCropper.getExtractionQuality(result).score}/100\`);
});

// 3. Save high-quality card:
await HighResCardExtractor.saveCard(
  cropResults[0], 
  'high_res_card.jpg',
  'image/jpeg',
  0.95  // High quality
);
  `);
}

/**
 * Performance comparison between standard and high-resolution extraction
 */
export function performanceComparison() {
  console.log('\n⚡ Performance Comparison:');
  console.log('========================');

  const scenarios = [
    {
      name: 'Mobile Camera (720p)',
      modelInput: { width: 1024, height: 768 },
      nativeRes: { width: 1280, height: 720 },
      improvement: '1.2x'
    },
    {
      name: 'HD Camera (1080p)',
      modelInput: { width: 1024, height: 768 },
      nativeRes: { width: 1920, height: 1080 },
      improvement: '2.6x'
    },
    {
      name: '4K Camera (2160p)',
      modelInput: { width: 1024, height: 768 },
      nativeRes: { width: 3840, height: 2160 },
      improvement: '10.5x'
    }
  ];

  scenarios.forEach(scenario => {
    const modelPixels = scenario.modelInput.width * scenario.modelInput.height;
    const nativePixels = scenario.nativeRes.width * scenario.nativeRes.height;
    const actualImprovement = (nativePixels / modelPixels).toFixed(1);
    
    console.log(`${scenario.name}:`);
    console.log(`  Model: ${scenario.modelInput.width}x${scenario.modelInput.height} (${(modelPixels / 1000000).toFixed(1)}MP)`);
    console.log(`  Native: ${scenario.nativeRes.width}x${scenario.nativeRes.height} (${(nativePixels / 1000000).toFixed(1)}MP)`);
    console.log(`  Improvement: ${actualImprovement}x more pixels`);
    console.log('');
  });

  console.log('💡 Key Benefits:');
  console.log('  ✅ Sharper text and details');
  console.log('  ✅ Better OCR accuracy potential');
  console.log('  ✅ Higher quality for archival');
  console.log('  ✅ Perspective correction for rotated cards');
  console.log('  ✅ Quality metrics for best result selection');
}

/**
 * Run the complete demo
 */
export async function runHighResDemo() {
  await demonstrateHighResExtraction();
  await exampleUsage();
  performanceComparison();
  
  console.log('\n🎉 High-Resolution Card Extraction Demo Complete!');
  console.log('Ready to extract cards at native camera resolution.');
}
