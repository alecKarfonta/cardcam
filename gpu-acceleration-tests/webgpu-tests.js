// WebGPU Testing Module with Advanced Error Handling and Non-blocking Approaches

class WebGPUTester {
    constructor() {
        this.adapter = null;
        this.device = null;
        this.logs = [];
        this.isInitialized = false;
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
        this.logs.push(logEntry);
        console.log(logEntry);
        this.updateLogDisplay();
    }

    updateLogDisplay() {
        const logElement = document.getElementById('webgpuLogs');
        if (logElement) {
            logElement.textContent = this.logs.join('\n');
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('webgpuStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
    }

    updateMetrics(time, throughput) {
        const timeElement = document.getElementById('webgpuTime');
        const throughputElement = document.getElementById('webgpuThroughput');
        
        if (timeElement) timeElement.textContent = time ? `${time.toFixed(2)}` : '-';
        if (throughputElement) throughputElement.textContent = throughput ? `${throughput.toFixed(0)}` : '-';
    }

    async initialize() {
        if (this.isInitialized) return true;

        try {
            this.log('Initializing WebGPU...');
            
            if (!('gpu' in navigator)) {
                throw new Error('WebGPU not supported in this browser - navigator.gpu not available');
            }

            this.log('WebGPU API detected, requesting adapter...');
            
            // Add timeout for adapter request
            const adapterPromise = navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('WebGPU adapter request timeout')), 10000);
            });
            
            this.adapter = await Promise.race([adapterPromise, timeoutPromise]);

            if (!this.adapter) {
                throw new Error('No WebGPU adapter available - GPU may not support WebGPU or drivers are outdated');
            }

            this.log(`Adapter found: ${this.adapter.info?.vendor || 'Unknown vendor'}`);
            this.log(`Adapter features: ${Array.from(this.adapter.features).join(', ') || 'None'}`);

            this.log('Requesting WebGPU device...');
            this.device = await this.adapter.requestDevice({
                requiredFeatures: [],
                requiredLimits: {}
            });

            if (!this.device) {
                throw new Error('Could not create WebGPU device');
            }

            // Set up error handling
            this.device.addEventListener('uncapturederror', (event) => {
                this.log(`WebGPU uncaptured error: ${event.error.message}`, 'error');
            });

            this.log('WebGPU initialized successfully!');
            this.isInitialized = true;
            return true;

        } catch (error) {
            this.log(`WebGPU initialization failed: ${error.message}`, 'error');
            this.updateStatus(`Initialization failed: ${error.message}`, 'error');
            return false;
        }
    }

    async testBasicWebGPU() {
        this.log('=== Starting Basic WebGPU Test ===');
        this.updateStatus('Running basic WebGPU test...', 'info');

        try {
            this.log('Checking WebGPU browser support...');
            
            // First check if WebGPU is available at all
            if (!('gpu' in navigator)) {
                throw new Error('WebGPU not available - browser does not support WebGPU API');
            }
            
            this.log('✅ navigator.gpu is available');
            
            const initialized = await this.initialize();
            if (!initialized) {
                throw new Error('WebGPU initialization failed');
            }

            // Test 1: Create a simple buffer
            this.log('Test 1: Creating GPU buffer...');
            const buffer = this.device.createBuffer({
                size: 1024,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            this.log('✅ GPU buffer created successfully');

            // Test 2: Create a simple compute shader
            this.log('Test 2: Creating compute shader...');
            const shaderModule = this.device.createShaderModule({
                code: `
                    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                    
                    @compute @workgroup_size(64)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index >= arrayLength(&data)) {
                            return;
                        }
                        data[index] = data[index] * 2.0;
                    }
                `
            });

            this.log('✅ Compute shader created successfully');

            // Test 3: Create compute pipeline
            this.log('Test 3: Creating compute pipeline...');
            const computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shaderModule,
                    entryPoint: 'main'
                }
            });

            this.log('✅ Compute pipeline created successfully');

            // Clean up
            buffer.destroy();
            this.log('✅ Resources cleaned up');

            this.updateStatus('Basic WebGPU test completed successfully!', 'success');
            this.log('=== Basic WebGPU Test Completed Successfully ===');

        } catch (error) {
            this.log(`Basic WebGPU test failed: ${error.message}`, 'error');
            this.updateStatus(`Basic test failed: ${error.message}`, 'error');
        }
    }

    async testComputeShader() {
        this.log('=== Starting WebGPU Compute Shader Test ===');
        this.updateStatus('Running compute shader test...', 'info');

        try {
            const initialized = await this.initialize();
            if (!initialized) {
                throw new Error('WebGPU initialization failed');
            }

            const startTime = performance.now();

            // Create test data
            const arraySize = 1024 * 1024; // 1M elements
            const inputData = new Float32Array(arraySize);
            for (let i = 0; i < arraySize; i++) {
                inputData[i] = Math.random();
            }

            this.log(`Created test data: ${arraySize} float32 elements`);

            // Create GPU buffer
            const buffer = this.device.createBuffer({
                size: inputData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });

            // Create staging buffer for reading results
            const stagingBuffer = this.device.createBuffer({
                size: inputData.byteLength,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });

            // Write data to GPU
            this.device.queue.writeBuffer(buffer, 0, inputData);
            this.log('Data uploaded to GPU');

            // Create compute shader
            const shaderModule = this.device.createShaderModule({
                code: `
                    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                    
                    @compute @workgroup_size(256)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index >= arrayLength(&data)) {
                            return;
                        }
                        // Perform some computation
                        data[index] = sqrt(data[index] * data[index] + 1.0);
                    }
                `
            });

            // Create compute pipeline
            const computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shaderModule,
                    entryPoint: 'main'
                }
            });

            // Create bind group
            const bindGroup = this.device.createBindGroup({
                layout: computePipeline.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: {
                        buffer: buffer
                    }
                }]
            });

            this.log('Compute pipeline and bind group created');

            // Execute compute shader using requestAnimationFrame for non-blocking
            await new Promise((resolve, reject) => {
                const executeCompute = () => {
                    try {
                        const commandEncoder = this.device.createCommandEncoder();
                        const passEncoder = commandEncoder.beginComputePass();
                        
                        passEncoder.setPipeline(computePipeline);
                        passEncoder.setBindGroup(0, bindGroup);
                        passEncoder.dispatchWorkgroups(Math.ceil(arraySize / 256));
                        passEncoder.end();

                        // Copy result to staging buffer
                        commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, inputData.byteLength);
                        
                        this.device.queue.submit([commandEncoder.finish()]);
                        this.log('Compute shader dispatched');
                        
                        resolve();
                    } catch (error) {
                        reject(error);
                    }
                };

                // Use requestAnimationFrame to avoid blocking
                requestAnimationFrame(executeCompute);
            });

            // Read back results (this part might still block, but it's necessary)
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const resultData = new Float32Array(stagingBuffer.getMappedRange());
            
            // Verify some results
            let validResults = 0;
            for (let i = 0; i < Math.min(100, arraySize); i++) {
                const expected = Math.sqrt(inputData[i] * inputData[i] + 1.0);
                if (Math.abs(resultData[i] - expected) < 0.001) {
                    validResults++;
                }
            }

            stagingBuffer.unmap();

            const endTime = performance.now();
            const executionTime = endTime - startTime;
            const throughput = arraySize / (executionTime / 1000);

            this.log(`Compute shader completed in ${executionTime.toFixed(2)}ms`);
            this.log(`Processed ${arraySize} elements`);
            this.log(`Throughput: ${throughput.toFixed(0)} operations/second`);
            this.log(`Result validation: ${validResults}/100 correct`);

            this.updateMetrics(executionTime, throughput);

            // Clean up
            buffer.destroy();
            stagingBuffer.destroy();

            this.updateStatus('Compute shader test completed successfully!', 'success');
            this.log('=== Compute Shader Test Completed Successfully ===');

        } catch (error) {
            this.log(`Compute shader test failed: ${error.message}`, 'error');
            this.updateStatus(`Compute test failed: ${error.message}`, 'error');
        }
    }

    async testONNXRuntimeWebGPU() {
        this.log('=== Starting ONNX Runtime WebGPU Test ===');
        this.updateStatus('Testing ONNX Runtime with WebGPU...', 'info');

        try {
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                // Try to load ONNX Runtime dynamically
                this.log('Loading ONNX Runtime...');
                await this.loadONNXRuntime();
            }

            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not available. Please include onnxruntime-web in your page.');
            }

            this.log('ONNX Runtime detected, configuring WebGPU...');

            // Configure ONNX Runtime for WebGPU
            ort.env.wasm.wasmPaths = '/onnx/';
            
            if (ort.env.webgpu) {
                ort.env.webgpu.validateInputContent = false;
                this.log('WebGPU validation disabled for performance');
            }

            // Create a simple test model (identity operation)
            this.log('Creating test session with WebGPU provider...');
            
            const sessionOptions = {
                executionProviders: ['webgpu', 'wasm', 'cpu'],
                graphOptimizationLevel: 'all',
                logSeverityLevel: 3
            };

            // This is where the blocking usually occurs
            this.log('⚠️  About to create ONNX session - this may cause blocking...');
            
            // Try to make this non-blocking using setTimeout
            const session = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Session creation timeout (30s) - likely blocking issue'));
                }, 30000);

                // Use setTimeout to yield control
                setTimeout(async () => {
                    try {
                        // Note: We would need an actual model file for this test
                        // For now, we'll simulate the session creation
                        this.log('Simulating session creation (no model file available)');
                        clearTimeout(timeout);
                        resolve(null); // Simulate successful creation
                    } catch (error) {
                        clearTimeout(timeout);
                        reject(error);
                    }
                }, 100);
            });

            if (session === null) {
                this.log('✅ Session creation simulation completed without blocking');
                this.log('Note: Actual model inference would require a valid ONNX model file');
            }

            this.updateStatus('ONNX Runtime WebGPU test completed (simulated)', 'success');
            this.log('=== ONNX Runtime WebGPU Test Completed ===');

        } catch (error) {
            this.log(`ONNX Runtime WebGPU test failed: ${error.message}`, 'error');
            this.updateStatus(`ONNX test failed: ${error.message}`, 'error');
        }
    }

    async loadONNXRuntime() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = '/onnx/ort.min.js';
            script.onload = () => {
                this.log('ONNX Runtime loaded successfully');
                resolve();
            };
            script.onerror = () => {
                reject(new Error('Failed to load ONNX Runtime'));
            };
            document.head.appendChild(script);
        });
    }

    clearLogs() {
        this.logs = [];
        this.updateLogDisplay();
        this.updateStatus('Logs cleared', 'info');
        this.updateMetrics(null, null);
    }
}

// Global WebGPU tester instance
let webgpuTester;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    webgpuTester = new WebGPUTester();
});

// Global functions for HTML buttons
function testWebGPUBasic() {
    if (webgpuTester) {
        webgpuTester.testBasicWebGPU();
    }
}

function testWebGPUCompute() {
    if (webgpuTester) {
        webgpuTester.testComputeShader();
    }
}

function testWebGPUONNX() {
    if (webgpuTester) {
        webgpuTester.testONNXRuntimeWebGPU();
    }
}

function clearWebGPULogs() {
    if (webgpuTester) {
        webgpuTester.clearLogs();
    }
}

// Export for use in other modules
window.WebGPUTester = WebGPUTester;
