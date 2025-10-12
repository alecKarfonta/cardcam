
/**
 * Multi-Algorithm Object Tracking Library
 * 
 * Provides a unified interface for multiple tracking algorithms:
 * - SORT (Simple Online and Realtime Tracking)
 * - DeepSORT (SORT with appearance features)
 * - ByteTrack (Low confidence detection recovery)
 * - IoUTracker (Simple IoU-based tracking)
 * - CentroidTracker (Distance-based tracking)
 * 
 * Usage:
 *   const tracker = new ByteTracker(options);
 *   const trackedDetections = tracker.update(detections);
 */

// ============================================================================
// SHARED COMPONENTS
// ============================================================================

/**
 * Kalman Filter for 2D position and velocity tracking
 * State: [x, y, vx, vy] (position and velocity)
 */
class KalmanFilter {
    constructor(dt = 1) {
      this.dt = dt;
      this.x = [0, 0, 0, 0];
      
      this.F = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ];
      
      this.H = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
      ];
      
      const q = 0.01;
      this.Q = [
        [q, 0, 0, 0],
        [0, q, 0, 0],
        [0, 0, q, 0],
        [0, 0, 0, q]
      ];
      
      const r = 0.1;
      this.R = [
        [r, 0],
        [0, r]
      ];
      
      this.P = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ];
    }
    
    predict() {
      const newX = this.matmul(this.F, [[this.x[0]], [this.x[1]], [this.x[2]], [this.x[3]]]);
      this.x = [newX[0][0], newX[1][0], newX[2][0], newX[3][0]];
      
      const FP = this.matmul(this.F, this.P);
      const FPFT = this.matmul(FP, this.transpose(this.F));
      this.P = this.matadd(FPFT, this.Q);
      
      return [this.x[0], this.x[1]];
    }
    
    update(detections) {
      this.frameCount++;
      
      // If no tracks exist, create new ones
      if (this.tracks.length === 0) {
        detections.forEach(det => {
          this.tracks.push(new CentroidTrack(det));
        });
        return this.tracks.map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
      }
      
      // If no detections, increment disappeared count
      if (detections.length === 0) {
        this.tracks.forEach(track => track.incrementDisappeared());
        this.tracks = this.tracks.filter(track => track.disappeared <= this.maxDisappeared);
        return this.tracks.map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
      }
      
      // Compute distance matrix
      const distMatrix = [];
      for (let i = 0; i < detections.length; i++) {
        const row = [];
        for (let j = 0; j < this.tracks.length; j++) {
          row.push(TrackingUtils.computeEuclideanDistance(detections[i], this.tracks[j].state));
        }
        distMatrix.push(row);
      }
      
      // Match using minimum distance
      const usedDets = new Set();
      const usedTracks = new Set();
      
      const flatDists = [];
      for (let i = 0; i < distMatrix.length; i++) {
        for (let j = 0; j < distMatrix[i].length; j++) {
          flatDists.push({ detIdx: i, trackIdx: j, dist: distMatrix[i][j] });
        }
      }
      
      flatDists.sort((a, b) => a.dist - b.dist);
      
      for (const { detIdx, trackIdx, dist } of flatDists) {
        if (dist > this.maxDistance) break;
        if (usedDets.has(detIdx) || usedTracks.has(trackIdx)) continue;
        
        this.tracks[trackIdx].update(detections[detIdx]);
        usedDets.add(detIdx);
        usedTracks.add(trackIdx);
      }
      
      // Handle unmatched detections (create new tracks)
      detections.forEach((det, i) => {
        if (!usedDets.has(i)) {
          this.tracks.push(new CentroidTrack(det));
        }
      });
      
      // Handle unmatched tracks (increment disappeared)
      this.tracks.forEach((track, i) => {
        if (!usedTracks.has(i)) {
          track.incrementDisappeared();
        }
      });
      
      // Remove disappeared tracks
      this.tracks = this.tracks.filter(track => track.disappeared <= this.maxDisappeared);
      
      return this.tracks.map(track => ({
        ...track.state,
        trackId: track.id,
        hits: track.hits,
        age: track.age
      }));
    }
    
    reset() {
      this.tracks = [];
      this.frameCount = 0;
      CentroidTrack.nextId = 1;
    }
  }
  
  // ============================================================================
  // TRACKER FACTORY AND COMPARISON UTILITIES
  // ============================================================================
  
  class TrackerFactory {
    static create(type, options = {}) {
      switch (type.toLowerCase()) {
        case 'bytetrack':
          return new ByteTracker(options);
        case 'sort':
          return new SORTTracker(options);
        case 'deepsort':
          return new DeepSORTTracker(options);
        case 'iou':
          return new IoUTracker(options);
        case 'centroid':
          return new CentroidTracker(options);
        default:
          throw new Error(`Unknown tracker type: ${type}`);
      }
    }
    
    static getDefaultOptions(type) {
      const defaults = {
        bytetrack: {
          trackHighThresh: 0.6,
          trackLowThresh: 0.3,
          newTrackThresh: 0.7,
          matchThresh: 0.8,
          maxAge: 30,
          minHits: 3
        },
        sort: {
          maxAge: 1,
          minHits: 3,
          iouThreshold: 0.3
        },
        deepsort: {
          maxAge: 30,
          minHits: 3,
          iouThreshold: 0.3,
          maxCosineDistance: 0.2,
          nnBudget: 100
        },
        iou: {
          iouThreshold: 0.3,
          maxAge: 5,
          minHits: 1
        },
        centroid: {
          maxDisappeared: 5,
          maxDistance: 0.1
        }
      };
      
      return defaults[type.toLowerCase()] || {};
    }
    
    static getAvailableTrackers() {
      return [
        {
          name: 'ByteTrack',
          id: 'bytetrack',
          description: 'Advanced tracker with low-confidence detection recovery',
          bestFor: 'Handling occlusions and missed detections',
          speed: 'Fast',
          accuracy: 'High'
        },
        {
          name: 'SORT',
          id: 'sort',
          description: 'Simple Online and Realtime Tracking with Kalman filtering',
          bestFor: 'Real-time applications with minimal overhead',
          speed: 'Very Fast',
          accuracy: 'Medium'
        },
        {
          name: 'DeepSORT',
          id: 'deepsort',
          description: 'SORT enhanced with appearance features',
          bestFor: 'Long-term occlusions and re-identification',
          speed: 'Medium',
          accuracy: 'Very High'
        },
        {
          name: 'IoU Tracker',
          id: 'iou',
          description: 'Simple IoU-based matching without motion model',
          bestFor: 'Static cameras with slow-moving objects',
          speed: 'Very Fast',
          accuracy: 'Low'
        },
        {
          name: 'Centroid Tracker',
          id: 'centroid',
          description: 'Distance-based tracking using object centroids',
          bestFor: 'Simple scenarios with well-separated objects',
          speed: 'Very Fast',
          accuracy: 'Low'
        }
      ];
    }
  }
  
  // ============================================================================
  // EXAMPLE USAGE AND TESTING
  // ============================================================================
  
  /**
   * Example usage:
   * 
   * // Create a tracker
   * const tracker = TrackerFactory.create('bytetrack', {
   *   trackHighThresh: 0.6,
   *   maxAge: 30
   * });
   * 
   * // In your inference loop
   * async function runInference() {
   *   const detections = await detectObjects(); // Your detection code
   *   const trackedDetections = tracker.update(detections);
   *   drawDetections(trackedDetections);
   * }
   * 
   * // Switching trackers
   * const sortTracker = TrackerFactory.create('sort');
   * const deepsortTracker = TrackerFactory.create('deepsort');
   * 
   * // Reset tracker
   * tracker.reset();
   */
  
  /**
   * Benchmark utility for comparing trackers
   */
  class TrackerBenchmark {
    constructor() {
      this.results = {};
    }
    
    async runBenchmark(detectionSequence, trackerConfigs) {
      for (const config of trackerConfigs) {
        const tracker = TrackerFactory.create(config.type, config.options);
        const startTime = performance.now();
        
        let totalTracks = 0;
        let totalIDSwitches = 0;
        let prevTracks = new Map();
        
        for (const detections of detectionSequence) {
          const tracked = tracker.update(detections);
          totalTracks += tracked.length;
          
          // Count ID switches (simple heuristic)
          const currentTracks = new Map();
          tracked.forEach(t => {
            currentTracks.set(t.trackId, { x: t.x, y: t.y });
          });
          
          // Check for position jumps (potential ID switch)
          currentTracks.forEach((pos, id) => {
            if (prevTracks.has(id)) {
              const prevPos = prevTracks.get(id);
              const dist = Math.sqrt(
                Math.pow(pos.x - prevPos.x, 2) + 
                Math.pow(pos.y - prevPos.y, 2)
              );
              if (dist > 0.3) totalIDSwitches++;
            }
          });
          
          prevTracks = currentTracks;
        }
        
        const endTime = performance.now();
        const avgTimePerFrame = (endTime - startTime) / detectionSequence.length;
        
        this.results[config.type] = {
          avgTimePerFrame: avgTimePerFrame.toFixed(2),
          totalTracks,
          idSwitches: totalIDSwitches,
          fps: (1000 / avgTimePerFrame).toFixed(1)
        };
      }
      
      return this.results;
    }
    
    printResults() {
      console.log('Tracker Benchmark Results:');
      console.log('==========================');
      for (const [type, result] of Object.entries(this.results)) {
        console.log(`\n${type.toUpperCase()}:`);
        console.log(`  Avg Time/Frame: ${result.avgTimePerFrame}ms`);
        console.log(`  FPS: ${result.fps}`);
        console.log(`  Total Tracks: ${result.totalTracks}`);
        console.log(`  ID Switches: ${result.idSwitches}`);
      }
    }
  }
  
  // Export for use in other scripts
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      ByteTracker,
      SORTTracker,
      DeepSORTTracker,
      IoUTracker,
      CentroidTracker,
      TrackerFactory,
      TrackerBenchmark,
      KalmanFilter,
      TrackingUtils
    };
  }measurement) {
      const Hx = this.matmul(this.H, [[this.x[0]], [this.x[1]], [this.x[2]], [this.x[3]]]);
      const y = [
        [measurement[0] - Hx[0][0]],
        [measurement[1] - Hx[1][0]]
      ];
      
      const HP = this.matmul(this.H, this.P);
      const HPHT = this.matmul(HP, this.transpose(this.H));
      const S = this.matadd(HPHT, this.R);
      
      const PHT = this.matmul(this.P, this.transpose(this.H));
      const Sinv = this.invert2x2(S);
      const K = this.matmul(PHT, Sinv);
      
      const Ky = this.matmul(K, y);
      this.x[0] += Ky[0][0];
      this.x[1] += Ky[1][0];
      this.x[2] += Ky[2][0];
      this.x[3] += Ky[3][0];
      
      const KH = this.matmul(K, this.H);
      const I_KH = [
        [1 - KH[0][0], -KH[0][1], -KH[0][2], -KH[0][3]],
        [-KH[1][0], 1 - KH[1][1], -KH[1][2], -KH[1][3]],
        [-KH[2][0], -KH[2][1], 1 - KH[2][2], -KH[2][3]],
        [-KH[3][0], -KH[3][1], -KH[3][2], 1 - KH[3][3]]
      ];
      this.P = this.matmul(I_KH, this.P);
      
      return [this.x[0], this.x[1]];
    }
    
    matmul(A, B) {
      const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
      for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
          for (let k = 0; k < B.length; k++) {
            result[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return result;
    }
    
    matadd(A, B) {
      return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    }
    
    transpose(A) {
      return A[0].map((_, i) => A.map(row => row[i]));
    }
    
    invert2x2(M) {
      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      return [
        [M[1][1] / det, -M[0][1] / det],
        [-M[1][0] / det, M[0][0] / det]
      ];
    }
  }
  
  /**
   * Utility functions for tracking algorithms
   */
  class TrackingUtils {
    static computeIoU(det1, det2) {
      const x1_min = det1.x - det1.width / 2;
      const y1_min = det1.y - det1.height / 2;
      const x1_max = det1.x + det1.width / 2;
      const y1_max = det1.y + det1.height / 2;
      
      const x2_min = det2.x - det2.width / 2;
      const y2_min = det2.y - det2.height / 2;
      const x2_max = det2.x + det2.width / 2;
      const y2_max = det2.y + det2.height / 2;
      
      const xi_min = Math.max(x1_min, x2_min);
      const yi_min = Math.max(y1_min, y2_min);
      const xi_max = Math.min(x1_max, x2_max);
      const yi_max = Math.min(y1_max, y2_max);
      
      const inter_area = Math.max(0, xi_max - xi_min) * Math.max(0, yi_max - yi_min);
      const area1 = (x1_max - x1_min) * (y1_max - y1_min);
      const area2 = (x2_max - x2_min) * (y2_max - y2_min);
      const union_area = area1 + area2 - inter_area;
      
      return union_area > 0 ? inter_area / union_area : 0;
    }
    
    static computeEuclideanDistance(det1, det2) {
      const dx = det1.x - det2.x;
      const dy = det1.y - det2.y;
      return Math.sqrt(dx * dx + dy * dy);
    }
    
    static generateAppearanceFeature(detection) {
      // Simplified appearance feature (in real implementation, use CNN embeddings)
      const seed = detection.x * 1000 + detection.y * 1000;
      const feature = [];
      for (let i = 0; i < 128; i++) {
        feature.push(Math.sin(seed + i) * Math.cos(seed * 2 + i));
      }
      return feature;
    }
    
    static cosineDistance(feat1, feat2) {
      let dot = 0, norm1 = 0, norm2 = 0;
      for (let i = 0; i < feat1.length; i++) {
        dot += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
      }
      return 1 - (dot / (Math.sqrt(norm1) * Math.sqrt(norm2)));
    }
  }
  
  // ============================================================================
  // BYTETRACK IMPLEMENTATION
  // ============================================================================
  
  class ByteTrack {
    static nextId = 1;
    
    constructor(detection) {
      this.id = ByteTrack.nextId++;
      this.kf = new KalmanFilter();
      this.kf.x = [detection.x, detection.y, 0, 0];
      this.hits = 1;
      this.age = 1;
      this.timeSinceUpdate = 0;
      this.state = detection;
      this.history = [];
    }
    
    predict() {
      const [x, y] = this.kf.predict();
      this.state = { ...this.state, x, y };
      this.age++;
      this.timeSinceUpdate++;
      return this.state;
    }
    
    update(detection) {
      this.kf.update([detection.x, detection.y]);
      this.state = detection;
      this.hits++;
      this.timeSinceUpdate = 0;
      this.history.push({ x: detection.x, y: detection.y });
      if (this.history.length > 30) this.history.shift();
    }
  }
  
  class ByteTracker {
    constructor(options = {}) {
      this.tracks = [];
      this.frameCount = 0;
      this.trackHighThresh = options.trackHighThresh || 0.6;
      this.trackLowThresh = options.trackLowThresh || 0.3;
      this.newTrackThresh = options.newTrackThresh || 0.7;
      this.matchThresh = options.matchThresh || 0.8;
      this.maxAge = options.maxAge || 30;
      this.minHits = options.minHits || 3;
    }
    
    update(detections) {
      this.frameCount++;
      
      const highDets = detections.filter(d => d.confidence >= this.trackHighThresh);
      const lowDets = detections.filter(d => 
        d.confidence >= this.trackLowThresh && d.confidence < this.trackHighThresh
      );
      
      this.tracks.forEach(track => track.predict());
      
      const unmatched1 = this.associateDetectionsToTracks(highDets, this.tracks);
      const unmatchedTracks = this.tracks.filter(t => t.timeSinceUpdate > 0);
      this.associateDetectionsToTracks(lowDets, unmatchedTracks);
      
      unmatched1.detections.forEach(det => {
        if (det.confidence >= this.newTrackThresh) {
          this.tracks.push(new ByteTrack(det));
        }
      });
      
      this.tracks = this.tracks.filter(track => track.timeSinceUpdate <= this.maxAge);
      
      return this.tracks
        .filter(track => track.hits >= this.minHits || this.frameCount <= this.minHits)
        .map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
    }
    
    associateDetectionsToTracks(detections, tracks) {
      if (detections.length === 0 || tracks.length === 0) {
        return { detections, tracks };
      }
      
      const costMatrix = [];
      for (let i = 0; i < detections.length; i++) {
        const row = [];
        for (let j = 0; j < tracks.length; j++) {
          row.push(TrackingUtils.computeIoU(detections[i], tracks[j].state));
        }
        costMatrix.push(row);
      }
      
      const matches = this.linearAssignment(costMatrix);
      
      const unmatchedDets = detections.filter((_, i) => 
        !matches.some(m => m.detIdx === i)
      );
      const unmatchedTracks = tracks.filter((_, i) => 
        !matches.some(m => m.trackIdx === i)
      );
      
      matches.forEach(({ detIdx, trackIdx }) => {
        tracks[trackIdx].update(detections[detIdx]);
      });
      
      return { detections: unmatchedDets, tracks: unmatchedTracks };
    }
    
    linearAssignment(costMatrix) {
      const matches = [];
      const usedDets = new Set();
      const usedTracks = new Set();
      
      const flatCosts = [];
      for (let i = 0; i < costMatrix.length; i++) {
        for (let j = 0; j < costMatrix[i].length; j++) {
          flatCosts.push({ detIdx: i, trackIdx: j, iou: costMatrix[i][j] });
        }
      }
      
      flatCosts.sort((a, b) => b.iou - a.iou);
      
      for (const { detIdx, trackIdx, iou } of flatCosts) {
        if (iou < this.matchThresh) break;
        if (usedDets.has(detIdx) || usedTracks.has(trackIdx)) continue;
        
        matches.push({ detIdx, trackIdx });
        usedDets.add(detIdx);
        usedTracks.add(trackIdx);
      }
      
      return matches;
    }
    
    reset() {
      this.tracks = [];
      this.frameCount = 0;
      ByteTrack.nextId = 1;
    }
  }
  
  // ============================================================================
  // SORT IMPLEMENTATION
  // ============================================================================
  
  class SORTTrack {
    static nextId = 1;
    
    constructor(detection) {
      this.id = SORTTrack.nextId++;
      this.kf = new KalmanFilter();
      this.kf.x = [detection.x, detection.y, 0, 0];
      this.hits = 1;
      this.age = 1;
      this.timeSinceUpdate = 0;
      this.state = detection;
    }
    
    predict() {
      const [x, y] = this.kf.predict();
      this.state = { ...this.state, x, y };
      this.age++;
      this.timeSinceUpdate++;
      return this.state;
    }
    
    update(detection) {
      this.kf.update([detection.x, detection.y]);
      this.state = detection;
      this.hits++;
      this.timeSinceUpdate = 0;
    }
  }
  
  class SORTTracker {
    constructor(options = {}) {
      this.tracks = [];
      this.frameCount = 0;
      this.maxAge = options.maxAge || 1;  // SORT typically uses 1
      this.minHits = options.minHits || 3;
      this.iouThreshold = options.iouThreshold || 0.3;
    }
    
    update(detections) {
      this.frameCount++;
      
      // Predict existing tracks
      this.tracks.forEach(track => track.predict());
      
      // Associate detections to tracks
      const { matched, unmatchedDets, unmatchedTracks } = 
        this.associateDetectionsToTracks(detections, this.tracks);
      
      // Update matched tracks
      matched.forEach(({ detIdx, trackIdx }) => {
        this.tracks[trackIdx].update(detections[detIdx]);
      });
      
      // Create new tracks for unmatched detections
      unmatchedDets.forEach(det => {
        this.tracks.push(new SORTTrack(det));
      });
      
      // Remove dead tracks
      this.tracks = this.tracks.filter(track => track.timeSinceUpdate <= this.maxAge);
      
      // Return confirmed tracks
      return this.tracks
        .filter(track => track.hits >= this.minHits || this.frameCount <= this.minHits)
        .map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
    }
    
    associateDetectionsToTracks(detections, tracks) {
      if (detections.length === 0 || tracks.length === 0) {
        return {
          matched: [],
          unmatchedDets: detections,
          unmatchedTracks: tracks
        };
      }
      
      const iouMatrix = [];
      for (let i = 0; i < detections.length; i++) {
        const row = [];
        for (let j = 0; j < tracks.length; j++) {
          row.push(TrackingUtils.computeIoU(detections[i], tracks[j].state));
        }
        iouMatrix.push(row);
      }
      
      const matched = [];
      const usedDets = new Set();
      const usedTracks = new Set();
      
      const flatCosts = [];
      for (let i = 0; i < iouMatrix.length; i++) {
        for (let j = 0; j < iouMatrix[i].length; j++) {
          flatCosts.push({ detIdx: i, trackIdx: j, iou: iouMatrix[i][j] });
        }
      }
      
      flatCosts.sort((a, b) => b.iou - a.iou);
      
      for (const { detIdx, trackIdx, iou } of flatCosts) {
        if (iou < this.iouThreshold) break;
        if (usedDets.has(detIdx) || usedTracks.has(trackIdx)) continue;
        
        matched.push({ detIdx, trackIdx });
        usedDets.add(detIdx);
        usedTracks.add(trackIdx);
      }
      
      const unmatchedDets = detections.filter((_, i) => !usedDets.has(i));
      const unmatchedTracks = tracks.filter((_, i) => !usedTracks.has(i));
      
      return { matched, unmatchedDets, unmatchedTracks };
    }
    
    reset() {
      this.tracks = [];
      this.frameCount = 0;
      SORTTrack.nextId = 1;
    }
  }
  
  // ============================================================================
  // DEEPSORT IMPLEMENTATION
  // ============================================================================
  
  class DeepSORTTrack {
    static nextId = 1;
    
    constructor(detection) {
      this.id = DeepSORTTrack.nextId++;
      this.kf = new KalmanFilter();
      this.kf.x = [detection.x, detection.y, 0, 0];
      this.hits = 1;
      this.age = 1;
      this.timeSinceUpdate = 0;
      this.state = detection;
      this.features = [TrackingUtils.generateAppearanceFeature(detection)];
      this.maxFeatures = 100;
    }
    
    predict() {
      const [x, y] = this.kf.predict();
      this.state = { ...this.state, x, y };
      this.age++;
      this.timeSinceUpdate++;
      return this.state;
    }
    
    update(detection) {
      this.kf.update([detection.x, detection.y]);
      this.state = detection;
      this.hits++;
      this.timeSinceUpdate = 0;
      
      // Update feature gallery
      this.features.push(TrackingUtils.generateAppearanceFeature(detection));
      if (this.features.length > this.maxFeatures) {
        this.features.shift();
      }
    }
    
    getAverageFeature() {
      const avgFeature = new Array(128).fill(0);
      for (const feat of this.features) {
        for (let i = 0; i < feat.length; i++) {
          avgFeature[i] += feat[i];
        }
      }
      return avgFeature.map(v => v / this.features.length);
    }
  }
  
  class DeepSORTTracker {
    constructor(options = {}) {
      this.tracks = [];
      this.frameCount = 0;
      this.maxAge = options.maxAge || 30;
      this.minHits = options.minHits || 3;
      this.iouThreshold = options.iouThreshold || 0.3;
      this.maxCosineDistance = options.maxCosineDistance || 0.2;
      this.nnBudget = options.nnBudget || 100;
    }
    
    update(detections) {
      this.frameCount++;
      
      // Predict existing tracks
      this.tracks.forEach(track => track.predict());
      
      // Associate detections to tracks using cascade matching
      const { matched, unmatchedDets, unmatchedTracks } = 
        this.cascadeMatching(detections, this.tracks);
      
      // Update matched tracks
      matched.forEach(({ detIdx, trackIdx }) => {
        this.tracks[trackIdx].update(detections[detIdx]);
      });
      
      // Create new tracks for unmatched detections
      unmatchedDets.forEach(det => {
        this.tracks.push(new DeepSORTTrack(det));
      });
      
      // Remove dead tracks
      this.tracks = this.tracks.filter(track => track.timeSinceUpdate <= this.maxAge);
      
      // Return confirmed tracks
      return this.tracks
        .filter(track => track.hits >= this.minHits || this.frameCount <= this.minHits)
        .map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
    }
    
    cascadeMatching(detections, tracks) {
      if (detections.length === 0 || tracks.length === 0) {
        return {
          matched: [],
          unmatchedDets: detections,
          unmatchedTracks: tracks
        };
      }
      
      const matched = [];
      let remainingDets = [...detections];
      const usedTracks = new Set();
      
      // Cascade matching by age
      for (let age = 0; age < this.maxAge; age++) {
        if (remainingDets.length === 0) break;
        
        const ageTracks = tracks.filter(t => 
          t.timeSinceUpdate === age && !usedTracks.has(t.id)
        );
        
        if (ageTracks.length === 0) continue;
        
        const ageMatches = this.matchDetectionsToTracks(remainingDets, ageTracks);
        
        ageMatches.forEach(({ detIdx, track }) => {
          matched.push({ 
            detIdx: detections.indexOf(remainingDets[detIdx]), 
            trackIdx: tracks.indexOf(track) 
          });
          usedTracks.add(track.id);
        });
        
        remainingDets = remainingDets.filter((_, i) => 
          !ageMatches.some(m => m.detIdx === i)
        );
      }
      
      const unmatchedTracks = tracks.filter(t => !usedTracks.has(t.id));
      
      return { matched, unmatchedDets: remainingDets, unmatchedTracks };
    }
    
    matchDetectionsToTracks(detections, tracks) {
      const matches = [];
      
      // Compute combined cost (appearance + motion)
      const costMatrix = [];
      for (let i = 0; i < detections.length; i++) {
        const row = [];
        for (let j = 0; j < tracks.length; j++) {
          const detFeature = TrackingUtils.generateAppearanceFeature(detections[i]);
          const trackFeature = tracks[j].getAverageFeature();
          const appearanceCost = TrackingUtils.cosineDistance(detFeature, trackFeature);
          const iou = TrackingUtils.computeIoU(detections[i], tracks[j].state);
          
          // Gate by appearance
          if (appearanceCost > this.maxCosineDistance) {
            row.push(-1); // Invalid match
          } else {
            row.push(iou); // Use IoU as final cost
          }
        }
        costMatrix.push(row);
      }
      
      // Hungarian assignment
      const usedDets = new Set();
      const usedTracks = new Set();
      
      const flatCosts = [];
      for (let i = 0; i < costMatrix.length; i++) {
        for (let j = 0; j < costMatrix[i].length; j++) {
          if (costMatrix[i][j] >= 0) {
            flatCosts.push({ detIdx: i, trackIdx: j, cost: costMatrix[i][j] });
          }
        }
      }
      
      flatCosts.sort((a, b) => b.cost - a.cost);
      
      for (const { detIdx, trackIdx, cost } of flatCosts) {
        if (cost < this.iouThreshold) break;
        if (usedDets.has(detIdx) || usedTracks.has(trackIdx)) continue;
        
        matches.push({ detIdx, track: tracks[trackIdx] });
        usedDets.add(detIdx);
        usedTracks.add(trackIdx);
      }
      
      return matches;
    }
    
    reset() {
      this.tracks = [];
      this.frameCount = 0;
      DeepSORTTrack.nextId = 1;
    }
  }
  
  // ============================================================================
  // IOU TRACKER IMPLEMENTATION
  // ============================================================================
  
  class IoUTrack {
    static nextId = 1;
    
    constructor(detection) {
      this.id = IoUTrack.nextId++;
      this.state = detection;
      this.timeSinceUpdate = 0;
      this.hits = 1;
      this.age = 1;
    }
    
    update(detection) {
      this.state = detection;
      this.hits++;
      this.timeSinceUpdate = 0;
    }
    
    incrementAge() {
      this.age++;
      this.timeSinceUpdate++;
    }
  }
  
  class IoUTracker {
    constructor(options = {}) {
      this.tracks = [];
      this.frameCount = 0;
      this.iouThreshold = options.iouThreshold || 0.3;
      this.maxAge = options.maxAge || 5;
      this.minHits = options.minHits || 1;
    }
    
    update(detections) {
      this.frameCount++;
      
      // Age existing tracks
      this.tracks.forEach(track => track.incrementAge());
      
      // Match detections to tracks
      if (this.tracks.length > 0 && detections.length > 0) {
        const iouMatrix = [];
        for (let i = 0; i < detections.length; i++) {
          const row = [];
          for (let j = 0; j < this.tracks.length; j++) {
            row.push(TrackingUtils.computeIoU(detections[i], this.tracks[j].state));
          }
          iouMatrix.push(row);
        }
        
        const matches = [];
        const usedDets = new Set();
        const usedTracks = new Set();
        
        const flatCosts = [];
        for (let i = 0; i < iouMatrix.length; i++) {
          for (let j = 0; j < iouMatrix[i].length; j++) {
            flatCosts.push({ detIdx: i, trackIdx: j, iou: iouMatrix[i][j] });
          }
        }
        
        flatCosts.sort((a, b) => b.iou - a.iou);
        
        for (const { detIdx, trackIdx, iou } of flatCosts) {
          if (iou < this.iouThreshold) break;
          if (usedDets.has(detIdx) || usedTracks.has(trackIdx)) continue;
          
          this.tracks[trackIdx].update(detections[detIdx]);
          usedDets.add(detIdx);
          usedTracks.add(trackIdx);
        }
        
        // Create new tracks for unmatched detections
        detections.forEach((det, i) => {
          if (!usedDets.has(i)) {
            this.tracks.push(new IoUTrack(det));
          }
        });
      } else if (detections.length > 0) {
        // No existing tracks, create new ones
        detections.forEach(det => {
          this.tracks.push(new IoUTrack(det));
        });
      }
      
      // Remove dead tracks
      this.tracks = this.tracks.filter(track => track.timeSinceUpdate <= this.maxAge);
      
      // Return confirmed tracks
      return this.tracks
        .filter(track => track.hits >= this.minHits)
        .map(track => ({
          ...track.state,
          trackId: track.id,
          hits: track.hits,
          age: track.age
        }));
    }
    
    reset() {
      this.tracks = [];
      this.frameCount = 0;
      IoUTrack.nextId = 1;
    }
  }
  
  // ============================================================================
  // CENTROID TRACKER IMPLEMENTATION
  // ============================================================================
  
  class CentroidTrack {
    static nextId = 1;
    
    constructor(detection) {
      this.id = CentroidTrack.nextId++;
      this.state = detection;
      this.disappeared = 0;
      this.hits = 1;
      this.age = 1;
    }
    
    update(detection) {
      this.state = detection;
      this.disappeared = 0;
      this.hits++;
    }
    
    incrementDisappeared() {
      this.disappeared++;
      this.age++;
    }
  }
  
  class CentroidTracker {
    constructor(options = {}) {
      this.tracks = [];
      this.frameCount = 0;
      this.maxDisappeared = options.maxDisappeared || 5;
      this.maxDistance = options.maxDistance || 0.1; // Normalized distance
    }
    
    update(