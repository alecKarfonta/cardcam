/**
 * ByteTrack Object Tracking Implementation
 * 
 * Usage:
 *   const tracker = new ByteTracker(options);
 *   const trackedDetections = tracker.update(detections);
 */

/**
 * Kalman Filter for 2D position and velocity tracking
 * State: [x, y, vx, vy] (position and velocity)
 */
class KalmanFilter {
  constructor(dt = 1) {
    this.dt = dt;
    // State: [x, y, vx, vy]
    this.x = [0, 0, 0, 0];
    
    // State transition matrix (constant velocity model)
    this.F = [
      [1, 0, dt, 0],
      [0, 1, 0, dt],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
    
    // Measurement matrix (we only measure position)
    this.H = [
      [1, 0, 0, 0],
      [0, 1, 0, 0]
    ];
    
    // Process noise covariance
    const q = 0.01;
    this.Q = [
      [q, 0, 0, 0],
      [0, q, 0, 0],
      [0, 0, q, 0],
      [0, 0, 0, q]
    ];
    
    // Measurement noise covariance
    const r = 0.1;
    this.R = [
      [r, 0],
      [0, r]
    ];
    
    // State covariance
    this.P = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
  }
  
  predict() {
    // x = F * x
    const newX = this.matmul(this.F, [[this.x[0]], [this.x[1]], [this.x[2]], [this.x[3]]]);
    this.x = [newX[0][0], newX[1][0], newX[2][0], newX[3][0]];
    
    // P = F * P * F^T + Q
    const FP = this.matmul(this.F, this.P);
    const FPFT = this.matmul(FP, this.transpose(this.F));
    this.P = this.matadd(FPFT, this.Q);
    
    return [this.x[0], this.x[1]];
  }
  
  update(measurement) {
    // y = z - H * x
    const Hx = this.matmul(this.H, [[this.x[0]], [this.x[1]], [this.x[2]], [this.x[3]]]);
    const y = [
      [measurement[0] - Hx[0][0]],
      [measurement[1] - Hx[1][0]]
    ];
    
    // S = H * P * H^T + R
    const HP = this.matmul(this.H, this.P);
    const HPHT = this.matmul(HP, this.transpose(this.H));
    const S = this.matadd(HPHT, this.R);
    
    // K = P * H^T * S^-1
    const PHT = this.matmul(this.P, this.transpose(this.H));
    const Sinv = this.invert2x2(S);
    const K = this.matmul(PHT, Sinv);
    
    // x = x + K * y
    const Ky = this.matmul(K, y);
    this.x[0] += Ky[0][0];
    this.x[1] += Ky[1][0];
    this.x[2] += Ky[2][0];
    this.x[3] += Ky[3][0];
    
    // P = (I - K * H) * P
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
  
  // Matrix operations
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
 * Track represents a single tracked object
 */
class Track {
  static nextId = 1;
  
  constructor(detection) {
    this.id = Track.nextId++;
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
    this.state = {
      ...this.state,
      x,
      y
    };
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

/**
 * ByteTrack Tracker Implementation
 */
class ByteTracker {
  constructor(options = {}) {
    this.tracks = [];
    this.frameCount = 0;
    
    // ByteTrack parameters
    this.trackHighThresh = options.trackHighThresh || 0.6;
    this.trackLowThresh = options.trackLowThresh || 0.3;
    this.newTrackThresh = options.newTrackThresh || 0.7;
    this.matchThresh = options.matchThresh || 0.8;
    this.maxAge = options.maxAge || 30;
    this.minHits = options.minHits || 3;
  }
  
  update(detections) {
    this.frameCount++;
    
    // Split detections into high and low confidence
    const highDets = detections.filter(d => d.confidence >= this.trackHighThresh);
    const lowDets = detections.filter(d => 
      d.confidence >= this.trackLowThresh && d.confidence < this.trackHighThresh
    );
    
    // Predict all tracks
    this.tracks.forEach(track => track.predict());
    
    // First association: high confidence detections
    const unmatched1 = this.associateDetectionsToTracks(highDets, this.tracks);
    
    // Second association: low confidence detections with unmatched tracks
    const unmatchedTracks = this.tracks.filter(t => t.timeSinceUpdate > 0);
    const unmatched2 = this.associateDetectionsToTracks(lowDets, unmatchedTracks);
    
    // Create new tracks from remaining high confidence detections
    unmatched1.detections.forEach(det => {
      if (det.confidence >= this.newTrackThresh) {
        this.tracks.push(new Track(det));
      }
    });
    
    // Remove dead tracks
    this.tracks = this.tracks.filter(track => 
      track.timeSinceUpdate <= this.maxAge
    );
    
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
      return { detections, tracks };
    }
    
    // Compute IoU cost matrix
    const costMatrix = this.computeIoUMatrix(detections, tracks);
    
    // Hungarian algorithm (greedy approximation)
    const matches = this.linearAssignment(costMatrix);
    
    const unmatchedDets = detections.filter((_, i) => 
      !matches.some(m => m.detIdx === i)
    );
    const unmatchedTracks = tracks.filter((_, i) => 
      !matches.some(m => m.trackIdx === i)
    );
    
    // Update matched tracks
    matches.forEach(({ detIdx, trackIdx }) => {
      tracks[trackIdx].update(detections[detIdx]);
    });
    
    return {
      detections: unmatchedDets,
      tracks: unmatchedTracks
    };
  }
  
  computeIoUMatrix(detections, tracks) {
    const matrix = [];
    for (let i = 0; i < detections.length; i++) {
      const row = [];
      for (let j = 0; j < tracks.length; j++) {
        row.push(this.computeIoU(detections[i], tracks[j].state));
      }
      matrix.push(row);
    }
    return matrix;
  }
  
  computeIoU(det1, det2) {
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
  
  linearAssignment(costMatrix) {
    const matches = [];
    const usedDets = new Set();
    const usedTracks = new Set();
    
    // Greedy matching (simplified Hungarian)
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
    Track.nextId = 1;
  }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ByteTracker, Track, KalmanFilter };
}

