/**
 * Dataset Management System for Computer Vision Training Data
 * 
 * This module provides a comprehensive system for capturing, managing, and editing
 * training examples for computer vision models. It follows object-oriented design
 * principles with clear separation of concerns.
 */

/**
 * Represents a single bounding box annotation
 */
class BoundingBox {
    constructor(x, y, width, height, rotation = 0, confidence = 1.0) {
        this.x = x;           // Center X (normalized 0-1)
        this.y = y;           // Center Y (normalized 0-1)
        this.width = width;   // Width (normalized 0-1)
        this.height = height; // Height (normalized 0-1)
        this.rotation = rotation; // Rotation in radians
        this.confidence = confidence;
        this.classId = 0;     // Default class ID
    }

    /**
     * Get the four corners of the rotated bounding box
     */
    getCorners() {
        const cos = Math.cos(this.rotation);
        const sin = Math.sin(this.rotation);
        const hw = this.width / 2;
        const hh = this.height / 2;

        const corners = [
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ];

        return corners.map(([px, py]) => {
            const rx = px * cos - py * sin;
            const ry = px * sin + py * cos;
            return [this.x + rx, this.y + ry];
        });
    }

    /**
     * Convert to YOLO OBB format (normalized coordinates)
     * Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
     */
    toYOLOOBB() {
        const corners = this.getCorners();
        const coords = corners.flat();
        return `${this.classId} ${coords.map(c => c.toFixed(6)).join(' ')}`;
    }

    /**
     * Create BoundingBox from YOLO OBB format
     */
    static fromYOLOOBB(line) {
        const parts = line.trim().split(/\s+/);
        if (parts.length !== 9) {
            throw new Error('Invalid YOLO OBB format');
        }

        const classId = parseInt(parts[0]);
        const coords = parts.slice(1).map(parseFloat);
        
        // Calculate center, width, height, and rotation from corners
        const x1 = coords[0], y1 = coords[1];
        const x2 = coords[2], y2 = coords[3];
        const x3 = coords[4], y3 = coords[5];
        const x4 = coords[6], y4 = coords[7];

        const centerX = (x1 + x2 + x3 + x4) / 4;
        const centerY = (y1 + y2 + y3 + y4) / 4;

        const width = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const height = Math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2);
        
        const rotation = Math.atan2(y2 - y1, x2 - x1);

        const bbox = new BoundingBox(centerX, centerY, width, height, rotation);
        bbox.classId = classId;
        return bbox;
    }

    /**
     * Clone this bounding box
     */
    clone() {
        const bbox = new BoundingBox(this.x, this.y, this.width, this.height, this.rotation, this.confidence);
        bbox.classId = this.classId;
        return bbox;
    }

    /**
     * Check if a point is inside this bounding box
     */
    containsPoint(px, py) {
        // Transform point to box's local coordinate system
        const cos = Math.cos(-this.rotation);
        const sin = Math.sin(-this.rotation);
        const dx = px - this.x;
        const dy = py - this.y;
        const localX = dx * cos - dy * sin;
        const localY = dx * sin + dy * cos;

        return Math.abs(localX) <= this.width / 2 && Math.abs(localY) <= this.height / 2;
    }
}

/**
 * Represents a single training example with image and annotations
 */
class TrainingExample {
    constructor(imageData, originalWidth, originalHeight) {
        this.id = TrainingExample.generateId();
        this.imageData = imageData; // Base64 encoded image data
        this.originalWidth = originalWidth;
        this.originalHeight = originalHeight;
        this.boundingBoxes = [];
        this.timestamp = new Date();
        this.metadata = {
            source: 'camera',
            captureDevice: null,
            lightingCondition: null,
            backgroundType: null
        };
    }

    static generateId() {
        return `example_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Add a bounding box to this example
     */
    addBoundingBox(bbox) {
        this.boundingBoxes.push(bbox);
    }

    /**
     * Remove a bounding box by index
     */
    removeBoundingBox(index) {
        if (index >= 0 && index < this.boundingBoxes.length) {
            this.boundingBoxes.splice(index, 1);
        }
    }

    /**
     * Update a bounding box at a specific index
     */
    updateBoundingBox(index, bbox) {
        if (index >= 0 && index < this.boundingBoxes.length) {
            this.boundingBoxes[index] = bbox;
        }
    }

    /**
     * Export as YOLO OBB format label file content
     */
    toYOLOLabels() {
        return this.boundingBoxes.map(bbox => bbox.toYOLOOBB()).join('\n');
    }

    /**
     * Get a summary of this example
     */
    getSummary() {
        return {
            id: this.id,
            timestamp: this.timestamp,
            dimensions: `${this.originalWidth}x${this.originalHeight}`,
            annotationCount: this.boundingBoxes.length,
            avgConfidence: this.boundingBoxes.length > 0
                ? (this.boundingBoxes.reduce((sum, b) => sum + b.confidence, 0) / this.boundingBoxes.length)
                : 0
        };
    }

    /**
     * Serialize to JSON (for storage)
     */
    toJSON() {
        return {
            id: this.id,
            imageData: this.imageData,
            originalWidth: this.originalWidth,
            originalHeight: this.originalHeight,
            boundingBoxes: this.boundingBoxes.map(bb => ({
                x: bb.x,
                y: bb.y,
                width: bb.width,
                height: bb.height,
                rotation: bb.rotation,
                confidence: bb.confidence,
                classId: bb.classId
            })),
            timestamp: this.timestamp.toISOString(),
            metadata: this.metadata
        };
    }

    /**
     * Deserialize from JSON
     */
    static fromJSON(json) {
        const example = new TrainingExample(json.imageData, json.originalWidth, json.originalHeight);
        example.id = json.id;
        example.timestamp = new Date(json.timestamp);
        example.metadata = json.metadata;
        example.boundingBoxes = json.boundingBoxes.map(bb =>
            new BoundingBox(bb.x, bb.y, bb.width, bb.height, bb.rotation, bb.confidence)
        );
        example.boundingBoxes.forEach((bbox, i) => {
            bbox.classId = json.boundingBoxes[i].classId;
        });
        return example;
    }
}

/**
 * Manages a collection of training examples
 */
class DatasetManager {
    constructor() {
        this.examples = [];
        this.currentExampleIndex = -1;
        this.storage = new DatasetStorage();
        this.storageInitialized = false;
        this.initStorage();
    }

    /**
     * Initialize storage system
     */
    async initStorage() {
        try {
            await this.storage.init();
            this.storageInitialized = true;
            await this.loadFromStorage();
        } catch (error) {
            console.error('Failed to initialize storage:', error);
        }
    }

    /**
     * Add a new training example
     */
    async addExample(example) {
        this.examples.push(example);
        await this.saveToStorage();
        return example.id;
    }

    /**
     * Remove an example by ID
     */
    async removeExample(exampleId) {
        const index = this.examples.findIndex(ex => ex.id === exampleId);
        if (index !== -1) {
            this.examples.splice(index, 1);
            await this.saveToStorage();
            return true;
        }
        return false;
    }

    /**
     * Get an example by ID
     */
    getExample(exampleId) {
        return this.examples.find(ex => ex.id === exampleId);
    }

    /**
     * Get all examples
     */
    getAllExamples() {
        return this.examples;
    }

    /**
     * Get example by index
     */
    getExampleByIndex(index) {
        if (index >= 0 && index < this.examples.length) {
            return this.examples[index];
        }
        return null;
    }

    /**
     * Get dataset statistics
     */
    getStats() {
        const totalExamples = this.examples.length;
        const totalAnnotations = this.examples.reduce((sum, ex) => sum + ex.boundingBoxes.length, 0);
        const avgAnnotationsPerExample = totalExamples > 0 ? totalAnnotations / totalExamples : 0;

        return {
            totalExamples,
            totalAnnotations,
            avgAnnotationsPerExample: avgAnnotationsPerExample.toFixed(2),
            storageSize: this.estimateStorageSize()
        };
    }

    /**
     * Estimate storage size in MB
     */
    estimateStorageSize() {
        try {
            const jsonStr = JSON.stringify(this.examples.map(ex => ex.toJSON()));
            return (jsonStr.length / (1024 * 1024)).toFixed(2) + ' MB';
        } catch (e) {
            return 'Unknown';
        }
    }

    /**
     * Clear all examples
     */
    async clearAll() {
        this.examples = [];
        this.currentExampleIndex = -1;
        await this.storage.clearAll();
    }

    /**
     * Save dataset to storage (IndexedDB or localStorage)
     */
    async saveToStorage() {
        if (!this.storageInitialized) {
            console.warn('Storage not initialized yet, waiting...');
            await this.initStorage();
        }

        try {
            const success = await this.storage.saveAll(this.examples);
            if (success) {
                const stats = await this.storage.getStats();
                console.log(`Saved to ${stats.storageType}: ${stats.exampleCount} examples, ${stats.estimatedSizeMB} MB`);
            }
            return success;
        } catch (e) {
            console.error('Failed to save to storage:', e);
            alert(`Storage error: ${e.message}`);
            return false;
        }
    }

    /**
     * Load dataset from storage (IndexedDB or localStorage)
     */
    async loadFromStorage() {
        try {
            const data = await this.storage.loadAll();
            this.examples = data.map(ex => TrainingExample.fromJSON(ex));
            console.log(`Loaded ${this.examples.length} examples from storage`);
        } catch (e) {
            console.error('Failed to load from storage:', e);
        }
    }

    /**
     * Get storage statistics
     */
    async getStorageStats() {
        return await this.storage.getStats();
    }

    /**
     * Export entire dataset as a ZIP file (conceptual - requires JSZip library)
     * For now, we'll export as individual files
     */
    async exportDataset(format = 'yolo') {
        const exports = [];

        for (let i = 0; i < this.examples.length; i++) {
            const example = this.examples[i];
            const baseName = `${format}_${i.toString().padStart(6, '0')}`;

            exports.push({
                filename: `${baseName}.jpg`,
                data: example.imageData,
                type: 'image'
            });

            exports.push({
                filename: `${baseName}.txt`,
                data: example.toYOLOLabels(),
                type: 'text'
            });
        }

        return exports;
    }

    /**
     * Create a training example from current camera frame
     */
    static async createFromVideoFrame(videoElement, detections, confidenceThreshold = 0.5) {
        // Create a canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);

        // Get base64 image data
        const imageData = canvas.toDataURL('image/jpeg', 0.95);

        // Create training example
        const example = new TrainingExample(
            imageData,
            videoElement.videoWidth,
            videoElement.videoHeight
        );

        // Add filtered detections as bounding boxes
        const filteredDetections = detections.filter(d => d.confidence >= confidenceThreshold);
        for (const det of filteredDetections) {
            // Detections are already in normalized coordinates (0-1)
            // DO NOT divide by videoWidth/videoHeight again!
            const bbox = new BoundingBox(
                det.x,
                det.y,
                det.width,
                det.height,
                det.angle || det.rotation || 0,
                det.confidence
            );
            bbox.classId = det.classId || 0;
            example.addBoundingBox(bbox);
        }

        return example;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BoundingBox, TrainingExample, DatasetManager };
}

