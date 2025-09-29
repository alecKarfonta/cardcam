// System Information Detection and Display

class SystemInfo {
    constructor() {
        this.info = {};
        this.init();
    }

    async init() {
        await this.detectSystemInfo();
        this.displaySystemInfo();
    }

    async detectSystemInfo() {
        // Browser Information
        this.info.browser = this.getBrowserInfo();
        
        // CPU Information
        this.info.cpuCores = navigator.hardwareConcurrency || 'Unknown';
        
        // Memory Information (if available)
        this.info.memory = this.getMemoryInfo();
        
        // GPU Information
        this.info.gpu = await this.getGPUInfo();
        
        // WebGPU Support
        this.info.webgpuSupport = await this.checkWebGPUSupport();
        
        // WebNN Support
        this.info.webnnSupport = await this.checkWebNNSupport();
        
        // Cross-Origin Isolation Status
        this.info.crossOriginIsolated = this.checkCrossOriginIsolation();
        
        console.log('System Information Detected:', this.info);
    }

    getBrowserInfo() {
        const ua = navigator.userAgent;
        let browser = 'Unknown';
        let version = 'Unknown';

        if (ua.includes('Chrome') && !ua.includes('Edg')) {
            browser = 'Chrome';
            const match = ua.match(/Chrome\/(\d+)/);
            version = match ? match[1] : 'Unknown';
        } else if (ua.includes('Firefox')) {
            browser = 'Firefox';
            const match = ua.match(/Firefox\/(\d+)/);
            version = match ? match[1] : 'Unknown';
        } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
            browser = 'Safari';
            const match = ua.match(/Version\/(\d+)/);
            version = match ? match[1] : 'Unknown';
        } else if (ua.includes('Edg')) {
            browser = 'Edge';
            const match = ua.match(/Edg\/(\d+)/);
            version = match ? match[1] : 'Unknown';
        }

        return `${browser} ${version}`;
    }

    getMemoryInfo() {
        if ('memory' in performance) {
            const memory = performance.memory;
            return {
                used: Math.round(memory.usedJSHeapSize / 1024 / 1024) + ' MB',
                total: Math.round(memory.totalJSHeapSize / 1024 / 1024) + ' MB',
                limit: Math.round(memory.jsHeapSizeLimit / 1024 / 1024) + ' MB'
            };
        }
        return 'Not available';
    }

    async getGPUInfo() {
        try {
            // Try WebGL first for basic GPU info
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (gl) {
                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                if (debugInfo) {
                    const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
                    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                    return `${vendor} - ${renderer}`;
                }
                return 'WebGL supported (GPU info masked)';
            }

            // Try WebGPU for more detailed info
            if ('gpu' in navigator) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const info = await adapter.requestAdapterInfo();
                    return `${info.vendor || 'Unknown'} - ${info.device || 'Unknown'}`;
                }
            }

            return 'No GPU information available';
        } catch (error) {
            console.error('Error getting GPU info:', error);
            return 'Error detecting GPU';
        }
    }

    async checkWebGPUSupport() {
        try {
            if (!('gpu' in navigator)) {
                return { supported: false, reason: 'navigator.gpu not available' };
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { supported: false, reason: 'No WebGPU adapter available' };
            }

            const device = await adapter.requestDevice();
            if (!device) {
                return { supported: false, reason: 'Could not create WebGPU device' };
            }

            // Test basic compute capability
            const computeSupported = adapter.features.has('shader-f16') || 
                                   adapter.features.has('timestamp-query') || 
                                   true; // Basic compute is always supported if device is created

            return { 
                supported: true, 
                features: Array.from(adapter.features),
                limits: {
                    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
                    maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
                    maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
                    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
                }
            };
        } catch (error) {
            return { supported: false, reason: error.message };
        }
    }

    async checkWebNNSupport() {
        try {
            if (!('ml' in navigator)) {
                return { supported: false, reason: 'navigator.ml not available' };
            }

            const context = await navigator.ml.createContext();
            if (!context) {
                return { supported: false, reason: 'Could not create WebNN context' };
            }

            // Test basic operation support
            const builder = new MLGraphBuilder(context);
            const operationSupport = {
                conv2d: typeof builder.conv2d === 'function',
                matmul: typeof builder.matmul === 'function',
                relu: typeof builder.relu === 'function',
                softmax: typeof builder.softmax === 'function'
            };

            return { 
                supported: true, 
                context: context.constructor.name,
                operations: operationSupport
            };
        } catch (error) {
            return { supported: false, reason: error.message };
        }
    }

    checkCrossOriginIsolation() {
        const isolated = crossOriginIsolated;
        const sharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
        
        return {
            isolated: isolated,
            sharedArrayBuffer: sharedArrayBuffer,
            status: isolated ? 'Enabled' : 'Disabled',
            reason: isolated ? 'Cross-origin isolation active' : 'Missing COOP/COEP headers'
        };
    }

    displaySystemInfo() {
        // Browser
        document.getElementById('browserInfo').textContent = this.info.browser;
        
        // GPU
        document.getElementById('gpuInfo').textContent = this.info.gpu;
        
        // CPU
        document.getElementById('cpuInfo').textContent = this.info.cpuCores;
        
        // Memory
        const memoryText = typeof this.info.memory === 'object' 
            ? `${this.info.memory.used} used / ${this.info.memory.limit} limit`
            : this.info.memory;
        document.getElementById('memoryInfo').textContent = memoryText;
        
        // WebGPU Support
        const webgpuText = this.info.webgpuSupport.supported 
            ? `✅ Supported (${this.info.webgpuSupport.features?.length || 0} features)`
            : `❌ Not supported: ${this.info.webgpuSupport.reason}`;
        document.getElementById('webgpuSupport').textContent = webgpuText;
        
        // WebNN Support
        const webnnText = this.info.webnnSupport.supported 
            ? `✅ Supported (${this.info.webnnSupport.context})`
            : `❌ Not supported: ${this.info.webnnSupport.reason}`;
        document.getElementById('webnnSupport').textContent = webnnText;
        
        // Cross-Origin Isolation
        const coiText = this.info.crossOriginIsolated.isolated 
            ? `✅ ${this.info.crossOriginIsolated.status}`
            : `❌ ${this.info.crossOriginIsolated.status} - ${this.info.crossOriginIsolated.reason}`;
        document.getElementById('coiStatus').textContent = coiText;
    }

    getSystemInfo() {
        return this.info;
    }
}

// Global system info instance
let systemInfo;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    systemInfo = new SystemInfo();
});

// Export for use in other modules
window.SystemInfo = SystemInfo;
