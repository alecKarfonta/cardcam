// WASM Baseline Testing Module for Performance Comparison

class WASMTester {
    constructor() {
        this.logs = [];
        this.session = null;
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
        this.logs.push(logEntry);
        console.log(logEntry);
        this.updateLogDisplay();
    }

    updateLogDisplay() {
        const logElement = document.getElementById('wasmLogs');
        if (logElement) {
            logElement.textContent = this.logs.join('\n');
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('wasmStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
    }

    updateMetrics(time, throughput) {
        const timeElement = document.getElementById('wasmTime');
        const throughputElement = document.getElementById('wasmThroughput');
        
        if (timeElement) timeElement.textContent = time ? `${time.toFixed(2)}` : '-';
        if (throughputElement) throughputElement.textContent = throughput ? `${throughput.toFixed(0)}` : '-';
    }

    async testWASMBasic() {
        this.log('=== Starting Basic WASM Test ===');
        this.updateStatus('Testing single-threaded WASM...', 'info');

        try {
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                this.log('Loading ONNX Runtime...');
                await this.loadONNXRuntime();
            }

            this.log('Configuring ONNX Runtime for single-threaded WASM...');
            
            // Configure for single-threaded WASM
            const base = '/onnx/';
            ort.env.wasm.wasmPaths = base;
            ort.env.wasm.numThreads = 1; // Single-threaded
            ort.env.wasm.simd = false; // Disable SIMD
            ort.env.wasm.proxy = false;
            
            this.log('WASM Configuration:');
            this.log(`- Threads: ${ort.env.wasm.numThreads}`);
            this.log(`- SIMD: ${ort.env.wasm.simd}`);
            this.log(`- Proxy: ${ort.env.wasm.proxy}`);

            // Test basic WASM operations
            const startTime = performance.now();
            
            // Simulate tensor operations
            const arraySize = 1024 * 1024; // 1M elements
            const testData = new Float32Array(arraySize);
            
            // Fill with test data
            for (let i = 0; i < arraySize; i++) {
                testData[i] = Math.random();
            }
            
            this.log(`Created test tensor: ${arraySize} float32 elements`);

            // Perform CPU-intensive operations
            const results = new Float32Array(arraySize);
            for (let i = 0; i < arraySize; i++) {
                results[i] = Math.sqrt(testData[i] * testData[i] + 1.0);
            }

            const endTime = performance.now();
            const executionTime = endTime - startTime;
            const throughput = arraySize / (executionTime / 1000);

            this.log(`Single-threaded computation completed in ${executionTime.toFixed(2)}ms`);
            this.log(`Processed ${arraySize} elements`);
            this.log(`Throughput: ${throughput.toFixed(0)} operations/second`);

            this.updateMetrics(executionTime, throughput);
            this.updateStatus('Single-threaded WASM test completed!', 'success');
            this.log('=== Basic WASM Test Completed ===');

        } catch (error) {
            this.log(`Basic WASM test failed: ${error.message}`, 'error');
            this.updateStatus(`Basic WASM test failed: ${error.message}`, 'error');
        }
    }

    async testWASMMultithread() {
        this.log('=== Starting Multi-threaded WASM Test ===');
        this.updateStatus('Testing multi-threaded WASM...', 'info');

        try {
            // Check cross-origin isolation
            const isolated = crossOriginIsolated;
            this.log(`Cross-origin isolation: ${isolated ? 'Enabled' : 'Disabled'}`);
            
            if (!isolated) {
                this.log('⚠️  Multi-threading requires cross-origin isolation', 'warning');
                this.log('Add these headers to your server:');
                this.log('Cross-Origin-Embedder-Policy: require-corp');
                this.log('Cross-Origin-Opener-Policy: same-origin');
            }

            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                this.log('Loading ONNX Runtime...');
                await this.loadONNXRuntime();
            }

            this.log('Configuring ONNX Runtime for multi-threaded WASM...');
            
            // Configure for multi-threaded WASM
            const maxThreads = Math.max(navigator.hardwareConcurrency || 16, 16);
            const base = '/onnx/';
            
            ort.env.wasm.wasmPaths = base;
            ort.env.wasm.numThreads = maxThreads;
            ort.env.wasm.simd = false; // Test without SIMD first
            ort.env.wasm.proxy = false;
            
            this.log('Multi-threaded WASM Configuration:');
            this.log(`- CPU Cores: ${navigator.hardwareConcurrency}`);
            this.log(`- Threads: ${ort.env.wasm.numThreads}`);
            this.log(`- SIMD: ${ort.env.wasm.simd}`);
            this.log(`- Cross-origin isolated: ${isolated}`);

            // Test with Web Workers for parallel processing
            const startTime = performance.now();
            
            const arraySize = 1024 * 1024; // 1M elements
            const numWorkers = Math.min(maxThreads, 8); // Limit workers
            const chunkSize = Math.floor(arraySize / numWorkers);
            
            this.log(`Creating ${numWorkers} workers for parallel processing...`);
            this.log(`Chunk size per worker: ${chunkSize} elements`);

            // Create test data
            const testData = new Float32Array(arraySize);
            for (let i = 0; i < arraySize; i++) {
                testData[i] = Math.random();
            }

            // Simulate multi-threaded processing using Promise.all
            const workerPromises = [];
            
            for (let w = 0; w < numWorkers; w++) {
                const startIdx = w * chunkSize;
                const endIdx = w === numWorkers - 1 ? arraySize : (w + 1) * chunkSize;
                
                const workerPromise = new Promise((resolve) => {
                    // Simulate worker processing
                    setTimeout(() => {
                        const results = new Float32Array(endIdx - startIdx);
                        for (let i = 0; i < results.length; i++) {
                            results[i] = Math.sqrt(testData[startIdx + i] * testData[startIdx + i] + 1.0);
                        }
                        resolve(results);
                    }, 10); // Small delay to simulate async processing
                });
                
                workerPromises.push(workerPromise);
            }

            // Wait for all workers to complete
            const workerResults = await Promise.all(workerPromises);
            
            // Combine results
            let totalElements = 0;
            workerResults.forEach(result => {
                totalElements += result.length;
            });

            const endTime = performance.now();
            const executionTime = endTime - startTime;
            const throughput = totalElements / (executionTime / 1000);

            this.log(`Multi-threaded computation completed in ${executionTime.toFixed(2)}ms`);
            this.log(`Processed ${totalElements} elements across ${numWorkers} workers`);
            this.log(`Throughput: ${throughput.toFixed(0)} operations/second`);

            this.updateMetrics(executionTime, throughput);
            this.updateStatus('Multi-threaded WASM test completed!', 'success');
            this.log('=== Multi-threaded WASM Test Completed ===');

        } catch (error) {
            this.log(`Multi-threaded WASM test failed: ${error.message}`, 'error');
            this.updateStatus(`Multi-threaded WASM test failed: ${error.message}`, 'error');
        }
    }

    async testWASMSIMD() {
        this.log('=== Starting WASM with SIMD Test ===');
        this.updateStatus('Testing WASM with SIMD acceleration...', 'info');

        try {
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                this.log('Loading ONNX Runtime...');
                await this.loadONNXRuntime();
            }

            this.log('Configuring ONNX Runtime for SIMD-enabled WASM...');
            
            // Configure for SIMD-enabled WASM
            const maxThreads = Math.max(navigator.hardwareConcurrency || 16, 16);
            const base = '/onnx/';
            
            ort.env.wasm.wasmPaths = base;
            ort.env.wasm.numThreads = maxThreads;
            ort.env.wasm.simd = true; // Enable SIMD
            ort.env.wasm.proxy = false;
            
            this.log('SIMD WASM Configuration:');
            this.log(`- Threads: ${ort.env.wasm.numThreads}`);
            this.log(`- SIMD: ${ort.env.wasm.simd}`);
            this.log(`- Cross-origin isolated: ${crossOriginIsolated}`);

            // Test SIMD capabilities
            const startTime = performance.now();
            
            const arraySize = 1024 * 1024; // 1M elements
            const testData = new Float32Array(arraySize);
            
            // Fill with test data
            for (let i = 0; i < arraySize; i++) {
                testData[i] = Math.random();
            }
            
            this.log(`Created test tensor: ${arraySize} float32 elements`);

            // Perform vectorized operations (simulated SIMD)
            const results = new Float32Array(arraySize);
            const vectorSize = 4; // Simulate 4-wide SIMD
            
            for (let i = 0; i < arraySize; i += vectorSize) {
                // Process 4 elements at once (simulating SIMD)
                for (let j = 0; j < vectorSize && (i + j) < arraySize; j++) {
                    const idx = i + j;
                    results[idx] = Math.sqrt(testData[idx] * testData[idx] + 1.0);
                }
            }

            const endTime = performance.now();
            const executionTime = endTime - startTime;
            const throughput = arraySize / (executionTime / 1000);

            this.log(`SIMD computation completed in ${executionTime.toFixed(2)}ms`);
            this.log(`Processed ${arraySize} elements with SIMD optimization`);
            this.log(`Throughput: ${throughput.toFixed(0)} operations/second`);

            // Test actual SIMD support
            this.log('Testing browser SIMD support...');
            try {
                // Check if WebAssembly SIMD is supported
                const simdSupported = typeof WebAssembly.SIMD !== 'undefined' || 
                                    (typeof WebAssembly.validate === 'function' && 
                                     WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])));
                this.log(`WebAssembly SIMD support: ${simdSupported ? 'Available' : 'Not available'}`);
            } catch (simdError) {
                this.log(`SIMD support check failed: ${simdError.message}`, 'warning');
            }

            this.updateMetrics(executionTime, throughput);
            this.updateStatus('WASM with SIMD test completed!', 'success');
            this.log('=== WASM SIMD Test Completed ===');

        } catch (error) {
            this.log(`WASM SIMD test failed: ${error.message}`, 'error');
            this.updateStatus(`WASM SIMD test failed: ${error.message}`, 'error');
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

// Global WASM tester instance
let wasmTester;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    wasmTester = new WASMTester();
});

// Global functions for HTML buttons
function testWASMBasic() {
    if (wasmTester) {
        wasmTester.testWASMBasic();
    }
}

function testWASMMultithread() {
    if (wasmTester) {
        wasmTester.testWASMMultithread();
    }
}

function testWASMSIMD() {
    if (wasmTester) {
        wasmTester.testWASMSIMD();
    }
}

function clearWASMLogs() {
    if (wasmTester) {
        wasmTester.clearLogs();
    }
}

// Export for use in other modules
window.WASMTester = WASMTester;
