// Comprehensive Benchmarking Module for GPU Acceleration Testing

class BenchmarkRunner {
    constructor() {
        this.results = [];
        this.isRunning = false;
        this.testModelGenerator = new TestModelGenerator();
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        const logEntry = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
        console.log(logEntry);
        this.updateBenchmarkDisplay(logEntry);
    }

    updateBenchmarkDisplay(message) {
        const logElement = document.getElementById('benchmarkResults');
        if (logElement) {
            logElement.textContent += message + '\n';
            logElement.scrollTop = logElement.scrollHeight;
        }
    }

    updateProgress(percentage) {
        const progressElement = document.getElementById('benchmarkProgress');
        if (progressElement) {
            progressElement.style.width = `${percentage}%`;
        }
    }

    async runFullBenchmark() {
        if (this.isRunning) {
            this.log('Benchmark already running...', 'warning');
            return;
        }

        this.isRunning = true;
        this.results = [];
        
        try {
            this.log('=== Starting Comprehensive GPU Acceleration Benchmark ===');
            this.log('This benchmark will test all available acceleration methods');
            this.log('');

            const totalTests = 12; // Adjust based on actual number of tests
            let completedTests = 0;

            // Test 1: System Information Gathering
            this.log('Phase 1: System Information Gathering');
            this.updateProgress((completedTests / totalTests) * 100);
            await this.gatherSystemInfo();
            completedTests++;

            // Test 2: WebGPU Capability Testing
            this.log('Phase 2: WebGPU Capability Testing');
            this.updateProgress((completedTests / totalTests) * 100);
            const webgpuResults = await this.benchmarkWebGPU();
            completedTests++;

            // Test 3: WebNN Capability Testing
            this.log('Phase 3: WebNN Capability Testing');
            this.updateProgress((completedTests / totalTests) * 100);
            const webnnResults = await this.benchmarkWebNN();
            completedTests++;

            // Test 4: WASM Baseline Testing
            this.log('Phase 4: WASM Baseline Testing');
            this.updateProgress((completedTests / totalTests) * 100);
            const wasmResults = await this.benchmarkWASM();
            completedTests++;

            // Test 5: Input Size Scaling
            this.log('Phase 5: Input Size Scaling Analysis');
            this.updateProgress((completedTests / totalTests) * 100);
            const scalingResults = await this.benchmarkInputSizeScaling();
            completedTests++;

            // Test 6: Memory Usage Analysis
            this.log('Phase 6: Memory Usage Analysis');
            this.updateProgress((completedTests / totalTests) * 100);
            const memoryResults = await this.benchmarkMemoryUsage();
            completedTests++;

            // Test 7: Concurrency Testing
            this.log('Phase 7: Concurrency Testing');
            this.updateProgress((completedTests / totalTests) * 100);
            const concurrencyResults = await this.benchmarkConcurrency();
            completedTests++;

            // Test 8: Stability Testing
            this.log('Phase 8: Stability Testing');
            this.updateProgress((completedTests / totalTests) * 100);
            const stabilityResults = await this.benchmarkStability();
            completedTests++;

            // Compile final results
            this.log('Phase 9: Compiling Results');
            this.updateProgress((completedTests / totalTests) * 100);
            const finalResults = this.compileResults({
                webgpu: webgpuResults,
                webnn: webnnResults,
                wasm: wasmResults,
                scaling: scalingResults,
                memory: memoryResults,
                concurrency: concurrencyResults,
                stability: stabilityResults
            });

            this.updateProgress(100);
            this.log('');
            this.log('=== BENCHMARK COMPLETED ===');
            this.displayFinalResults(finalResults);

        } catch (error) {
            this.log(`Benchmark failed: ${error.message}`, 'error');
        } finally {
            this.isRunning = false;
        }
    }

    async gatherSystemInfo() {
        this.log('Gathering system information...');
        
        const systemInfo = {
            browser: navigator.userAgent,
            cpuCores: navigator.hardwareConcurrency,
            memory: performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
            } : 'Not available',
            crossOriginIsolated: crossOriginIsolated,
            webgpuSupported: 'gpu' in navigator,
            webnnSupported: 'ml' in navigator,
            timestamp: new Date().toISOString()
        };

        this.log(`Browser: ${systemInfo.browser.split(' ')[0]}`);
        this.log(`CPU Cores: ${systemInfo.cpuCores}`);
        this.log(`Cross-origin Isolation: ${systemInfo.crossOriginIsolated}`);
        this.log(`WebGPU Support: ${systemInfo.webgpuSupported}`);
        this.log(`WebNN Support: ${systemInfo.webnnSupported}`);
        
        return systemInfo;
    }

    async benchmarkWebGPU() {
        this.log('Testing WebGPU performance...');
        
        const results = {
            supported: false,
            initTime: null,
            computeTime: null,
            throughput: null,
            error: null,
            stability: 'unknown'
        };

        try {
            if (!('gpu' in navigator)) {
                results.error = 'WebGPU not supported';
                return results;
            }

            const startInit = performance.now();
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                results.error = 'No WebGPU adapter available';
                return results;
            }

            const device = await adapter.requestDevice();
            const endInit = performance.now();
            
            results.supported = true;
            results.initTime = endInit - startInit;

            // Test compute performance
            const computeStart = performance.now();
            await this.testModelGenerator.simulateInference('webgpu');
            const computeEnd = performance.now();
            
            results.computeTime = computeEnd - computeStart;
            results.throughput = 150528 / (results.computeTime / 1000); // Standard input size
            results.stability = 'stable';

            this.log(`WebGPU - Init: ${results.initTime.toFixed(2)}ms, Compute: ${results.computeTime.toFixed(2)}ms`);

        } catch (error) {
            results.error = error.message;
            results.stability = 'unstable';
            this.log(`WebGPU test failed: ${error.message}`, 'error');
        }

        return results;
    }

    async benchmarkWebNN() {
        this.log('Testing WebNN performance...');
        
        const results = {
            supported: false,
            initTime: null,
            computeTime: null,
            throughput: null,
            error: null,
            operations: {}
        };

        try {
            if (!('ml' in navigator)) {
                results.error = 'WebNN not supported';
                return results;
            }

            const startInit = performance.now();
            const context = await navigator.ml.createContext();
            const endInit = performance.now();
            
            results.supported = true;
            results.initTime = endInit - startInit;

            // Test compute performance
            const computeStart = performance.now();
            await this.testModelGenerator.simulateInference('webnn');
            const computeEnd = performance.now();
            
            results.computeTime = computeEnd - computeStart;
            results.throughput = 150528 / (results.computeTime / 1000);

            // Test operation support
            try {
                const builder = new MLGraphBuilder(context);
                const operations = ['conv2d', 'matmul', 'relu', 'softmax', 'reshape'];
                for (const op of operations) {
                    results.operations[op] = typeof builder[op] === 'function';
                }
            } catch (builderError) {
                this.log(`MLGraphBuilder test failed: ${builderError.message}`, 'warning');
            }

            this.log(`WebNN - Init: ${results.initTime.toFixed(2)}ms, Compute: ${results.computeTime.toFixed(2)}ms`);

        } catch (error) {
            results.error = error.message;
            this.log(`WebNN test failed: ${error.message}`, 'error');
        }

        return results;
    }

    async benchmarkWASM() {
        this.log('Testing WASM performance...');
        
        const results = {
            singleThread: null,
            multiThread: null,
            simd: null
        };

        // Single-threaded WASM
        try {
            const singleStart = performance.now();
            await this.testModelGenerator.simulateInference('cpu');
            const singleEnd = performance.now();
            
            results.singleThread = {
                time: singleEnd - singleStart,
                throughput: 150528 / ((singleEnd - singleStart) / 1000)
            };
            
            this.log(`WASM Single-thread: ${results.singleThread.time.toFixed(2)}ms`);
        } catch (error) {
            this.log(`Single-thread WASM failed: ${error.message}`, 'error');
        }

        // Multi-threaded WASM
        try {
            const multiStart = performance.now();
            await this.testModelGenerator.simulateInference('wasm');
            const multiEnd = performance.now();
            
            results.multiThread = {
                time: multiEnd - multiStart,
                throughput: 150528 / ((multiEnd - multiStart) / 1000),
                threads: navigator.hardwareConcurrency
            };
            
            this.log(`WASM Multi-thread: ${results.multiThread.time.toFixed(2)}ms`);
        } catch (error) {
            this.log(`Multi-thread WASM failed: ${error.message}`, 'error');
        }

        return results;
    }

    async benchmarkInputSizeScaling() {
        this.log('Testing input size scaling...');
        
        const inputSizes = [
            { name: 'Small', shape: [1, 3, 32, 32] },
            { name: 'Medium', shape: [1, 3, 128, 128] },
            { name: 'Large', shape: [1, 3, 224, 224] },
            { name: 'XLarge', shape: [1, 3, 512, 512] }
        ];

        const results = {};
        
        for (const size of inputSizes) {
            this.log(`Testing ${size.name} input (${size.shape.join('x')})...`);
            
            try {
                const result = await this.testModelGenerator.simulateInference('wasm', size.shape);
                results[size.name] = {
                    shape: size.shape,
                    elements: size.shape.reduce((a, b) => a * b, 1),
                    time: result.executionTime,
                    throughput: result.throughput
                };
                
                this.log(`${size.name}: ${result.executionTime.toFixed(2)}ms`);
            } catch (error) {
                this.log(`${size.name} test failed: ${error.message}`, 'error');
            }
        }

        return results;
    }

    async benchmarkMemoryUsage() {
        this.log('Testing memory usage patterns...');
        
        const results = {
            baseline: null,
            peak: null,
            afterGC: null
        };

        if (performance.memory) {
            results.baseline = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
            
            // Allocate large arrays to test memory handling
            const largeArrays = [];
            for (let i = 0; i < 10; i++) {
                largeArrays.push(new Float32Array(1024 * 1024)); // 1M floats each
            }
            
            results.peak = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
            
            // Clear arrays and suggest GC
            largeArrays.length = 0;
            if (window.gc) {
                window.gc();
            }
            
            // Wait a bit for GC
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            results.afterGC = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
            
            this.log(`Memory - Baseline: ${results.baseline}MB, Peak: ${results.peak}MB, After GC: ${results.afterGC}MB`);
        } else {
            this.log('Memory API not available', 'warning');
        }

        return results;
    }

    async benchmarkConcurrency() {
        this.log('Testing concurrent inference...');
        
        const results = {
            sequential: null,
            concurrent: null,
            speedup: null
        };

        const numInferences = 5;

        // Sequential execution
        const seqStart = performance.now();
        for (let i = 0; i < numInferences; i++) {
            await this.testModelGenerator.simulateInference('wasm');
        }
        const seqEnd = performance.now();
        results.sequential = seqEnd - seqStart;

        // Concurrent execution
        const concStart = performance.now();
        const promises = [];
        for (let i = 0; i < numInferences; i++) {
            promises.push(this.testModelGenerator.simulateInference('wasm'));
        }
        await Promise.all(promises);
        const concEnd = performance.now();
        results.concurrent = concEnd - concStart;

        results.speedup = results.sequential / results.concurrent;

        this.log(`Concurrency - Sequential: ${results.sequential.toFixed(2)}ms, Concurrent: ${results.concurrent.toFixed(2)}ms`);
        this.log(`Speedup: ${results.speedup.toFixed(2)}x`);

        return results;
    }

    async benchmarkStability() {
        this.log('Testing stability over multiple runs...');
        
        const results = {
            runs: [],
            mean: null,
            stddev: null,
            stability: 'unknown'
        };

        const numRuns = 10;
        
        for (let i = 0; i < numRuns; i++) {
            try {
                const result = await this.testModelGenerator.simulateInference('wasm');
                results.runs.push(result.executionTime);
                this.log(`Run ${i + 1}: ${result.executionTime.toFixed(2)}ms`);
            } catch (error) {
                this.log(`Run ${i + 1} failed: ${error.message}`, 'error');
                results.runs.push(null);
            }
        }

        // Calculate statistics
        const validRuns = results.runs.filter(r => r !== null);
        if (validRuns.length > 0) {
            results.mean = validRuns.reduce((a, b) => a + b, 0) / validRuns.length;
            const variance = validRuns.reduce((a, b) => a + Math.pow(b - results.mean, 2), 0) / validRuns.length;
            results.stddev = Math.sqrt(variance);
            
            const cv = results.stddev / results.mean; // Coefficient of variation
            results.stability = cv < 0.1 ? 'stable' : cv < 0.2 ? 'moderate' : 'unstable';
            
            this.log(`Stability - Mean: ${results.mean.toFixed(2)}ms, StdDev: ${results.stddev.toFixed(2)}ms`);
            this.log(`Coefficient of Variation: ${(cv * 100).toFixed(1)}% (${results.stability})`);
        }

        return results;
    }

    compileResults(allResults) {
        const summary = {
            timestamp: new Date().toISOString(),
            recommendations: [],
            bestPerformer: null,
            issues: [],
            systemInfo: {
                browser: navigator.userAgent.split(' ')[0],
                cpuCores: navigator.hardwareConcurrency,
                crossOriginIsolated: crossOriginIsolated
            }
        };

        // Analyze WebGPU
        if (allResults.webgpu.supported && !allResults.webgpu.error) {
            summary.recommendations.push('✅ WebGPU is functional and may provide GPU acceleration');
        } else {
            summary.issues.push(`❌ WebGPU: ${allResults.webgpu.error || 'Not supported'}`);
        }

        // Analyze WebNN
        if (allResults.webnn.supported && !allResults.webnn.error) {
            summary.recommendations.push('✅ WebNN is functional and may provide hardware acceleration');
        } else {
            summary.issues.push(`❌ WebNN: ${allResults.webnn.error || 'Not supported'}`);
        }

        // Analyze WASM
        if (allResults.wasm.multiThread && allResults.wasm.singleThread) {
            const speedup = allResults.wasm.singleThread.time / allResults.wasm.multiThread.time;
            if (speedup > 1.5) {
                summary.recommendations.push(`✅ Multi-threaded WASM provides ${speedup.toFixed(1)}x speedup`);
            }
        }

        // Determine best performer
        const performers = [];
        if (allResults.webgpu.supported && allResults.webgpu.throughput) {
            performers.push({ name: 'WebGPU', throughput: allResults.webgpu.throughput });
        }
        if (allResults.webnn.supported && allResults.webnn.throughput) {
            performers.push({ name: 'WebNN', throughput: allResults.webnn.throughput });
        }
        if (allResults.wasm.multiThread) {
            performers.push({ name: 'Multi-threaded WASM', throughput: allResults.wasm.multiThread.throughput });
        }

        if (performers.length > 0) {
            summary.bestPerformer = performers.reduce((best, current) => 
                current.throughput > best.throughput ? current : best
            );
        }

        return summary;
    }

    displayFinalResults(results) {
        this.log('');
        this.log('=== FINAL BENCHMARK RESULTS ===');
        this.log('');
        
        this.log('SYSTEM INFORMATION:');
        this.log(`Browser: ${results.systemInfo.browser}`);
        this.log(`CPU Cores: ${results.systemInfo.cpuCores}`);
        this.log(`Cross-origin Isolation: ${results.systemInfo.crossOriginIsolated}`);
        this.log('');

        if (results.bestPerformer) {
            this.log('BEST PERFORMER:');
            this.log(`${results.bestPerformer.name}: ${results.bestPerformer.throughput.toFixed(0)} ops/sec`);
            this.log('');
        }

        if (results.recommendations.length > 0) {
            this.log('RECOMMENDATIONS:');
            results.recommendations.forEach(rec => this.log(rec));
            this.log('');
        }

        if (results.issues.length > 0) {
            this.log('ISSUES FOUND:');
            results.issues.forEach(issue => this.log(issue));
            this.log('');
        }

        this.log('Benchmark completed successfully!');
        this.log('Use "Export Results" to save detailed data.');
    }

    exportResults() {
        const exportData = {
            timestamp: new Date().toISOString(),
            systemInfo: systemInfo ? systemInfo.getSystemInfo() : null,
            results: this.results,
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `gpu-acceleration-benchmark-${Date.now()}.json`;
        link.click();
        
        this.log('Results exported successfully!');
    }

    clearAllResults() {
        this.results = [];
        const logElement = document.getElementById('benchmarkResults');
        if (logElement) {
            logElement.textContent = 'Benchmark results will appear here...\n';
        }
        this.updateProgress(0);
        this.log('All results cleared');
    }
}

// Global benchmark runner instance
let benchmarkRunner;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    benchmarkRunner = new BenchmarkRunner();
});

// Global functions for HTML buttons
function runFullBenchmark() {
    if (benchmarkRunner) {
        benchmarkRunner.runFullBenchmark();
    }
}

function exportResults() {
    if (benchmarkRunner) {
        benchmarkRunner.exportResults();
    }
}

function clearAllResults() {
    if (benchmarkRunner) {
        benchmarkRunner.clearAllResults();
    }
}

// Export for use in other modules
window.BenchmarkRunner = BenchmarkRunner;
