/**
 * Performance monitoring utility for tracking inference performance improvements
 */

export interface PerformanceMetrics {
  inferenceTime: number;
  preprocessingTime: number;
  postprocessingTime: number;
  totalTime: number;
  executionProvider: string;
  modelPath: string;
  inputSize: { width: number; height: number };
  timestamp: number;
}

export interface PerformanceStats {
  averageInferenceTime: number;
  medianInferenceTime: number;
  minInferenceTime: number;
  maxInferenceTime: number;
  standardDeviation: number;
  totalSamples: number;
  executionProviderStats: Record<string, {
    count: number;
    averageTime: number;
  }>;
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = [];
  private maxSamples = 100; // Keep last 100 measurements

  recordMetrics(metrics: PerformanceMetrics): void {
    this.metrics.push(metrics);
    
    // Keep only the most recent samples
    if (this.metrics.length > this.maxSamples) {
      this.metrics = this.metrics.slice(-this.maxSamples);
    }

    // Log performance info
    console.log(`⚡ Inference Performance: ${metrics.inferenceTime.toFixed(2)}ms (${metrics.executionProvider})`);
    console.log(`   - Preprocessing: ${metrics.preprocessingTime.toFixed(2)}ms`);
    console.log(`   - Postprocessing: ${metrics.postprocessingTime.toFixed(2)}ms`);
    console.log(`   - Total: ${metrics.totalTime.toFixed(2)}ms`);

    // Generate periodic reports every 10 samples
    if (this.metrics.length % 10 === 0) {
      console.log('\n📊 === PERFORMANCE REPORT ===');
      this.logPerformanceReport();
      console.log('=================================\n');
    }
  }

  getStats(): PerformanceStats {
    if (this.metrics.length === 0) {
      return {
        averageInferenceTime: 0,
        medianInferenceTime: 0,
        minInferenceTime: 0,
        maxInferenceTime: 0,
        standardDeviation: 0,
        totalSamples: 0,
        executionProviderStats: {}
      };
    }

    const inferenceTimes = this.metrics.map(m => m.inferenceTime);
    const sortedTimes = [...inferenceTimes].sort((a, b) => a - b);
    
    const average = inferenceTimes.reduce((sum, time) => sum + time, 0) / inferenceTimes.length;
    const median = sortedTimes[Math.floor(sortedTimes.length / 2)];
    const min = Math.min(...inferenceTimes);
    const max = Math.max(...inferenceTimes);
    
    // Calculate standard deviation
    const variance = inferenceTimes.reduce((sum, time) => sum + Math.pow(time - average, 2), 0) / inferenceTimes.length;
    const standardDeviation = Math.sqrt(variance);

    // Calculate execution provider stats
    const executionProviderStats: Record<string, { count: number; averageTime: number }> = {};
    
    for (const metric of this.metrics) {
      const provider = metric.executionProvider;
      if (!executionProviderStats[provider]) {
        executionProviderStats[provider] = { count: 0, averageTime: 0 };
      }
      executionProviderStats[provider].count++;
    }

    // Calculate average times for each provider
    for (const provider in executionProviderStats) {
      const providerMetrics = this.metrics.filter(m => m.executionProvider === provider);
      const providerTimes = providerMetrics.map(m => m.inferenceTime);
      executionProviderStats[provider].averageTime = 
        providerTimes.reduce((sum, time) => sum + time, 0) / providerTimes.length;
    }

    return {
      averageInferenceTime: average,
      medianInferenceTime: median,
      minInferenceTime: min,
      maxInferenceTime: max,
      standardDeviation,
      totalSamples: this.metrics.length,
      executionProviderStats
    };
  }

  logPerformanceReport(): void {
    const stats = this.getStats();
    
    if (stats.totalSamples === 0) {
      console.log('📊 No performance data available yet');
      return;
    }

    console.log('\n📊 Performance Report:');
    console.log(`   Samples: ${stats.totalSamples}`);
    console.log(`   Average: ${stats.averageInferenceTime.toFixed(2)}ms`);
    console.log(`   Median: ${stats.medianInferenceTime.toFixed(2)}ms`);
    console.log(`   Min: ${stats.minInferenceTime.toFixed(2)}ms`);
    console.log(`   Max: ${stats.maxInferenceTime.toFixed(2)}ms`);
    console.log(`   Std Dev: ${stats.standardDeviation.toFixed(2)}ms`);
    
    console.log('\n🔧 Execution Provider Performance:');
    for (const [provider, providerStats] of Object.entries(stats.executionProviderStats)) {
      console.log(`   ${provider}: ${providerStats.averageTime.toFixed(2)}ms avg (${providerStats.count} samples)`);
    }

    // Performance assessment
    if (stats.averageInferenceTime < 100) {
      console.log('🚀 Excellent performance! Under 100ms average');
    } else if (stats.averageInferenceTime < 200) {
      console.log('⚡ Good performance! Under 200ms average');
    } else if (stats.averageInferenceTime < 500) {
      console.log('⚠️ Moderate performance. Consider optimizations.');
    } else {
      console.log('🐌 Poor performance. Optimization needed.');
    }
  }

  getRecentTrend(samples: number = 10): { improving: boolean; trend: number } {
    if (this.metrics.length < samples * 2) {
      return { improving: false, trend: 0 };
    }

    const recent = this.metrics.slice(-samples);
    const previous = this.metrics.slice(-samples * 2, -samples);

    const recentAvg = recent.reduce((sum, m) => sum + m.inferenceTime, 0) / recent.length;
    const previousAvg = previous.reduce((sum, m) => sum + m.inferenceTime, 0) / previous.length;

    const trend = ((recentAvg - previousAvg) / previousAvg) * 100;
    const improving = trend < 0; // Negative trend means faster times

    return { improving, trend: Math.abs(trend) };
  }

  clear(): void {
    this.metrics = [];
    console.log('🗑️ Performance metrics cleared');
  }
}

// Global performance monitor instance
export const performanceMonitor = new PerformanceMonitor();

// Make it available globally for debugging
if (typeof window !== 'undefined') {
  (window as any).performanceMonitor = performanceMonitor;
}

// Utility function to measure execution time
export async function measurePerformance<T>(
  operation: () => Promise<T>,
  label: string
): Promise<{ result: T; duration: number }> {
  const startTime = performance.now();
  const result = await operation();
  const duration = performance.now() - startTime;
  
  console.log(`⏱️ ${label}: ${duration.toFixed(2)}ms`);
  
  return { result, duration };
}

// Utility to detect the best execution provider
export function detectOptimalExecutionProvider(): string[] {
  const providers: string[] = [];
  
  // Check WebGPU support
  if ('gpu' in navigator && navigator.gpu) {
    providers.push('webgpu');
    console.log('🚀 WebGPU detected - will provide best performance');
  }
  
  // Check WebGL support
  const canvas = document.createElement('canvas');
  const webgl2 = canvas.getContext('webgl2');
  const webgl = canvas.getContext('webgl');
  
  if (webgl2) {
    providers.push('webgl');
    console.log('⚡ WebGL 2.0 detected - good GPU acceleration available');
  } else if (webgl) {
    providers.push('webgl');
    console.log('⚡ WebGL 1.0 detected - basic GPU acceleration available');
  }
  
  // Always include WASM as fallback
  providers.push('wasm');
  
  // Check for SIMD support
  if (typeof WebAssembly !== 'undefined') {
    console.log('🔧 WebAssembly support detected');
  }
  
  console.log(`🎯 Optimal execution provider order: [${providers.join(', ')}]`);
  return providers;
}
