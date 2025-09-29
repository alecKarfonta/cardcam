// Simple Test Model Generator for GPU Acceleration Testing

class TestModelGenerator {
    constructor() {
        this.modelCache = new Map();
    }

    // Generate a simple ONNX model programmatically (for testing purposes)
    generateSimpleModel() {
        // This creates a simple mathematical model that can test GPU acceleration
        // without requiring external model files
        
        const modelData = {
            name: 'SimpleTestModel',
            description: 'Basic mathematical operations for GPU testing',
            inputShape: [1, 3, 224, 224], // Standard image input shape
            outputShape: [1, 1000], // Classification-like output
            operations: [
                'conv2d',
                'relu',
                'maxpool',
                'flatten',
                'matmul',
                'softmax'
            ]
        };

        return modelData;
    }

    // Create test tensors for inference
    createTestTensors(inputShape = [1, 3, 224, 224]) {
        const totalElements = inputShape.reduce((a, b) => a * b, 1);
        const inputData = new Float32Array(totalElements);
        
        // Fill with realistic image-like data (normalized 0-1)
        for (let i = 0; i < totalElements; i++) {
            inputData[i] = Math.random();
        }

        return {
            inputData,
            inputShape,
            totalElements
        };
    }

    // Simulate model inference for performance testing
    async simulateInference(provider = 'cpu', inputShape = [1, 3, 224, 224]) {
        const startTime = performance.now();
        
        const { inputData, totalElements } = this.createTestTensors(inputShape);
        
        // Simulate different types of operations based on provider
        let results;
        
        switch (provider) {
            case 'webgpu':
                results = await this.simulateWebGPUInference(inputData);
                break;
            case 'webnn':
                results = await this.simulateWebNNInference(inputData);
                break;
            case 'wasm':
                results = await this.simulateWASMInference(inputData);
                break;
            default:
                results = await this.simulateCPUInference(inputData);
        }
        
        const endTime = performance.now();
        const executionTime = endTime - startTime;
        
        return {
            provider,
            executionTime,
            inputShape,
            totalElements,
            throughput: totalElements / (executionTime / 1000),
            results: results.slice(0, 10) // Return first 10 results for verification
        };
    }

    async simulateWebGPUInference(inputData) {
        // Simulate WebGPU compute operations
        await this.delay(50); // Simulate GPU setup time
        
        const results = new Float32Array(inputData.length);
        
        // Simulate parallel GPU processing (vectorized operations)
        for (let i = 0; i < inputData.length; i += 4) {
            // Simulate SIMD-like parallel processing
            for (let j = 0; j < 4 && (i + j) < inputData.length; j++) {
                const idx = i + j;
                results[idx] = Math.tanh(inputData[idx] * 2.0 + 0.5);
            }
        }
        
        return results;
    }

    async simulateWebNNInference(inputData) {
        // Simulate WebNN operations
        await this.delay(30); // Simulate WebNN setup time
        
        const results = new Float32Array(inputData.length);
        
        // Simulate optimized neural network operations
        for (let i = 0; i < inputData.length; i++) {
            results[i] = 1.0 / (1.0 + Math.exp(-inputData[i])); // Sigmoid activation
        }
        
        return results;
    }

    async simulateWASMInference(inputData) {
        // Simulate WASM operations
        await this.delay(10); // Simulate WASM setup time
        
        const results = new Float32Array(inputData.length);
        
        // Simulate multi-threaded WASM processing
        const numThreads = navigator.hardwareConcurrency || 4;
        const chunkSize = Math.floor(inputData.length / numThreads);
        
        const promises = [];
        for (let t = 0; t < numThreads; t++) {
            const start = t * chunkSize;
            const end = t === numThreads - 1 ? inputData.length : (t + 1) * chunkSize;
            
            promises.push(this.processChunk(inputData, results, start, end));
        }
        
        await Promise.all(promises);
        return results;
    }

    async simulateCPUInference(inputData) {
        // Simulate single-threaded CPU operations
        await this.delay(5); // Minimal setup time
        
        const results = new Float32Array(inputData.length);
        
        // Simple sequential processing
        for (let i = 0; i < inputData.length; i++) {
            results[i] = Math.sqrt(inputData[i] * inputData[i] + 1.0);
        }
        
        return results;
    }

    async processChunk(inputData, results, start, end) {
        return new Promise((resolve) => {
            setTimeout(() => {
                for (let i = start; i < end; i++) {
                    results[i] = Math.sqrt(inputData[i] * inputData[i] + 1.0);
                }
                resolve();
            }, 1);
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Create a more complex test scenario
    createComplexTestScenario() {
        return {
            name: 'Complex Neural Network Simulation',
            layers: [
                { type: 'conv2d', inputShape: [1, 3, 224, 224], outputShape: [1, 64, 112, 112] },
                { type: 'relu', inputShape: [1, 64, 112, 112], outputShape: [1, 64, 112, 112] },
                { type: 'maxpool', inputShape: [1, 64, 112, 112], outputShape: [1, 64, 56, 56] },
                { type: 'conv2d', inputShape: [1, 64, 56, 56], outputShape: [1, 128, 28, 28] },
                { type: 'relu', inputShape: [1, 128, 28, 28], outputShape: [1, 128, 28, 28] },
                { type: 'flatten', inputShape: [1, 128, 28, 28], outputShape: [1, 100352] },
                { type: 'matmul', inputShape: [1, 100352], outputShape: [1, 1000] },
                { type: 'softmax', inputShape: [1, 1000], outputShape: [1, 1000] }
            ],
            totalOperations: 100352 * 1000 + 64 * 112 * 112 + 128 * 28 * 28,
            estimatedFLOPs: 2.5e9 // 2.5 billion floating point operations
        };
    }

    // Benchmark different input sizes
    async benchmarkInputSizes(provider = 'cpu') {
        const inputSizes = [
            [1, 3, 32, 32],    // Small: 3,072 elements
            [1, 3, 64, 64],    // Medium: 12,288 elements  
            [1, 3, 128, 128],  // Large: 49,152 elements
            [1, 3, 224, 224],  // Standard: 150,528 elements
            [1, 3, 512, 512],  // Very Large: 786,432 elements
        ];

        const results = [];
        
        for (const inputShape of inputSizes) {
            const result = await this.simulateInference(provider, inputShape);
            results.push(result);
        }
        
        return results;
    }
}

// Export for use in other modules
window.TestModelGenerator = TestModelGenerator;
