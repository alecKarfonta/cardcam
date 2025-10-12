/**
 * Interactive Bounding Box Editor
 * 
 * Provides an interactive canvas-based editor for adjusting bounding boxes
 * in training examples. Supports moving, resizing, and rotating bounding boxes.
 */

class BoundingBoxEditor {
    constructor(canvasElement, imageElement) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.imageElement = imageElement;
        
        this.trainingExample = null;
        this.selectedBboxIndex = -1;
        this.hoveredBboxIndex = -1;
        this.activeHandle = null;
        
        // Interaction state
        this.isDragging = false;
        this.isResizing = false;
        this.isRotating = false;
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.dragStartBbox = null;
        
        // Image display bounds (for letterboxing)
        this.imageDrawX = 0;
        this.imageDrawY = 0;
        this.imageDrawWidth = 0;
        this.imageDrawHeight = 0;
        
        // Visual settings
        this.colors = {
            default: '#00ff00',
            selected: '#ff00ff',
            hovered: '#ffff00',
            handle: '#ffffff'
        };
        this.handleSize = 8;
        this.rotationHandleDistance = 40;
        
        // Callbacks
        this.onBboxChanged = null;
        this.onBboxSelected = null;
        
        this.setupEventListeners();
    }

    /**
     * Load a training example for editing
     */
    loadExample(example) {
        this.trainingExample = example;
        this.selectedBboxIndex = -1;
        this.hoveredBboxIndex = -1;
        this.render();
    }

    /**
     * Setup mouse event listeners
     */
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));
    }

    /**
     * Convert canvas coordinates to normalized image coordinates
     * Accounts for letterboxing
     */
    canvasToNormalized(canvasX, canvasY) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        // Convert client coordinates to canvas coordinates
        const canvasPixelX = (canvasX - rect.left) * scaleX;
        const canvasPixelY = (canvasY - rect.top) * scaleY;
        
        // Account for letterboxing offset
        const imageLocalX = canvasPixelX - this.imageDrawX;
        const imageLocalY = canvasPixelY - this.imageDrawY;
        
        // Normalize relative to image dimensions
        const x = imageLocalX / this.imageDrawWidth;
        const y = imageLocalY / this.imageDrawHeight;
        
        return { x, y };
    }

    /**
     * Convert normalized coordinates to canvas coordinates
     * Accounts for letterboxing
     */
    normalizedToCanvas(normX, normY) {
        return {
            x: normX * this.imageDrawWidth + this.imageDrawX,
            y: normY * this.imageDrawHeight + this.imageDrawY
        };
    }

    /**
     * Find which bounding box contains a point
     */
    findBboxAtPoint(normX, normY) {
        if (!this.trainingExample) return -1;

        // Check in reverse order (top to bottom in z-order)
        for (let i = this.trainingExample.boundingBoxes.length - 1; i >= 0; i--) {
            const bbox = this.trainingExample.boundingBoxes[i];
            if (bbox.containsPoint(normX, normY)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find which handle is at a point
     * Returns: { type: 'corner'|'edge'|'rotation', index: number } or null
     */
    findHandleAtPoint(normX, normY) {
        if (this.selectedBboxIndex === -1) return null;

        const bbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex];
        const corners = bbox.getCorners();
        const handleSize = this.handleSize / this.canvas.width;

        // Check corner handles
        for (let i = 0; i < corners.length; i++) {
            const [cx, cy] = corners[i];
            const dist = Math.sqrt((normX - cx) ** 2 + (normY - cy) ** 2);
            if (dist < handleSize) {
                return { type: 'corner', index: i };
            }
        }

        // Check rotation handle (above the top edge)
        const { x: cx, y: cy } = this.normalizedToCanvas(bbox.x, bbox.y);
        const rotHandleX = cx;
        const rotHandleY = cy - this.rotationHandleDistance;
        const { x: rotNormX, y: rotNormY } = this.canvasToNormalized(
            rotHandleX + this.canvas.getBoundingClientRect().left,
            rotHandleY + this.canvas.getBoundingClientRect().top
        );
        const rotDist = Math.sqrt((normX - rotNormX) ** 2 + (normY - rotNormY) ** 2);
        if (rotDist < handleSize) {
            return { type: 'rotation', index: 0 };
        }

        return null;
    }

    /**
     * Handle mouse down event
     */
    handleMouseDown(e) {
        if (!this.trainingExample) return;

        const { x, y } = this.canvasToNormalized(e.clientX, e.clientY);
        
        // Check if clicking on a handle
        const handle = this.findHandleAtPoint(x, y);
        if (handle) {
            if (handle.type === 'rotation') {
                this.isRotating = true;
            } else {
                this.isResizing = true;
            }
            this.activeHandle = handle;
            this.dragStartX = x;
            this.dragStartY = y;
            this.dragStartBbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex].clone();
            return;
        }

        // Check if clicking on a bounding box
        const bboxIndex = this.findBboxAtPoint(x, y);
        if (bboxIndex !== -1) {
            this.selectedBboxIndex = bboxIndex;
            this.isDragging = true;
            this.dragStartX = x;
            this.dragStartY = y;
            this.dragStartBbox = this.trainingExample.boundingBoxes[bboxIndex].clone();
            
            if (this.onBboxSelected) {
                this.onBboxSelected(bboxIndex);
            }
            
            this.render();
        } else {
            this.selectedBboxIndex = -1;
            this.render();
        }
    }

    /**
     * Handle mouse move event
     */
    handleMouseMove(e) {
        if (!this.trainingExample) return;

        const { x, y } = this.canvasToNormalized(e.clientX, e.clientY);

        if (this.isDragging && this.selectedBboxIndex !== -1) {
            // Move the bounding box
            const dx = x - this.dragStartX;
            const dy = y - this.dragStartY;
            
            const bbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex];
            bbox.x = this.dragStartBbox.x + dx;
            bbox.y = this.dragStartBbox.y + dy;
            
            // Clamp to image bounds
            bbox.x = Math.max(bbox.width / 2, Math.min(1 - bbox.width / 2, bbox.x));
            bbox.y = Math.max(bbox.height / 2, Math.min(1 - bbox.height / 2, bbox.y));
            
            this.render();
            
            if (this.onBboxChanged) {
                this.onBboxChanged(this.selectedBboxIndex, bbox);
            }
        } else if (this.isResizing && this.activeHandle && this.selectedBboxIndex !== -1) {
            // Resize the bounding box
            this.handleResize(x, y);
        } else if (this.isRotating && this.selectedBboxIndex !== -1) {
            // Rotate the bounding box
            this.handleRotation(x, y);
        } else {
            // Update hover state
            const handle = this.findHandleAtPoint(x, y);
            if (handle) {
                this.canvas.style.cursor = handle.type === 'rotation' ? 'grab' : 'nwse-resize';
            } else {
                const bboxIndex = this.findBboxAtPoint(x, y);
                this.hoveredBboxIndex = bboxIndex;
                this.canvas.style.cursor = bboxIndex !== -1 ? 'move' : 'default';
                this.render();
            }
        }
    }

    /**
     * Handle resize operation
     */
    handleResize(x, y) {
        const bbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex];
        const handle = this.activeHandle;

        // Transform to bbox local coordinates
        const cos = Math.cos(-bbox.rotation);
        const sin = Math.sin(-bbox.rotation);
        
        const dx = x - bbox.x;
        const dy = y - bbox.y;
        const localX = dx * cos - dy * sin;
        const localY = dx * sin + dy * cos;

        // Determine new dimensions based on which corner
        const newWidth = Math.abs(localX) * 2;
        const newHeight = Math.abs(localY) * 2;

        // Apply minimum size constraint
        bbox.width = Math.max(0.02, Math.min(1.0, newWidth));
        bbox.height = Math.max(0.02, Math.min(1.0, newHeight));

        this.render();

        if (this.onBboxChanged) {
            this.onBboxChanged(this.selectedBboxIndex, bbox);
        }
    }

    /**
     * Handle rotation operation
     */
    handleRotation(x, y) {
        const bbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex];
        
        // Calculate angle from bbox center to mouse
        const dx = x - bbox.x;
        const dy = y - bbox.y;
        bbox.rotation = Math.atan2(dy, dx) + Math.PI / 2;

        this.render();

        if (this.onBboxChanged) {
            this.onBboxChanged(this.selectedBboxIndex, bbox);
        }
    }

    /**
     * Handle mouse up event
     */
    handleMouseUp(e) {
        this.isDragging = false;
        this.isResizing = false;
        this.isRotating = false;
        this.activeHandle = null;
        this.canvas.style.cursor = 'default';
    }

    /**
     * Handle mouse leave event
     */
    handleMouseLeave(e) {
        this.handleMouseUp(e);
        this.hoveredBboxIndex = -1;
        this.render();
    }

    /**
     * Handle mouse wheel for fine rotation adjustment
     */
    handleWheel(e) {
        if (this.selectedBboxIndex === -1) return;
        
        e.preventDefault();
        
        const bbox = this.trainingExample.boundingBoxes[this.selectedBboxIndex];
        const rotationStep = Math.PI / 180; // 1 degree
        bbox.rotation += e.deltaY > 0 ? rotationStep : -rotationStep;
        
        this.render();

        if (this.onBboxChanged) {
            this.onBboxChanged(this.selectedBboxIndex, bbox);
        }
    }

    /**
     * Calculate letterbox dimensions for image
     */
    calculateImageBounds() {
        if (!this.imageElement || !this.imageElement.complete) {
            this.imageDrawX = 0;
            this.imageDrawY = 0;
            this.imageDrawWidth = this.canvas.width;
            this.imageDrawHeight = this.canvas.height;
            return;
        }

        const imgAspect = this.imageElement.naturalWidth / this.imageElement.naturalHeight;
        const canvasAspect = this.canvas.width / this.canvas.height;

        if (imgAspect > canvasAspect) {
            // Image is wider - fit to width
            this.imageDrawWidth = this.canvas.width;
            this.imageDrawHeight = this.canvas.width / imgAspect;
            this.imageDrawX = 0;
            this.imageDrawY = (this.canvas.height - this.imageDrawHeight) / 2;
        } else {
            // Image is taller - fit to height
            this.imageDrawHeight = this.canvas.height;
            this.imageDrawWidth = this.canvas.height * imgAspect;
            this.imageDrawY = 0;
            this.imageDrawX = (this.canvas.width - this.imageDrawWidth) / 2;
        }
    }

    /**
     * Render the editor
     */
    render() {
        if (!this.trainingExample) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }

        // Clear canvas with black background
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Calculate image bounds for letterboxing
        this.calculateImageBounds();

        // Draw image with proper aspect ratio (letterboxed)
        if (this.imageElement && this.imageElement.complete) {
            this.ctx.drawImage(
                this.imageElement,
                this.imageDrawX,
                this.imageDrawY,
                this.imageDrawWidth,
                this.imageDrawHeight
            );
        }

        // Draw all bounding boxes
        this.trainingExample.boundingBoxes.forEach((bbox, index) => {
            const isSelected = index === this.selectedBboxIndex;
            const isHovered = index === this.hoveredBboxIndex;
            
            let color = this.colors.default;
            if (isSelected) color = this.colors.selected;
            else if (isHovered) color = this.colors.hovered;

            this.drawBoundingBox(bbox, color, isSelected);
        });
    }

    /**
     * Draw a single bounding box
     */
    drawBoundingBox(bbox, color, drawHandles = false) {
        const corners = bbox.getCorners();
        const canvasCorners = corners.map(([x, y]) => this.normalizedToCanvas(x, y));

        this.ctx.save();

        // Draw box
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(canvasCorners[0].x, canvasCorners[0].y);
        for (let i = 1; i < canvasCorners.length; i++) {
            this.ctx.lineTo(canvasCorners[i].x, canvasCorners[i].y);
        }
        this.ctx.closePath();
        this.ctx.stroke();

        // Draw center point
        const { x: cx, y: cy } = this.normalizedToCanvas(bbox.x, bbox.y);
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw confidence label
        this.ctx.fillStyle = color;
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 3;
        this.ctx.font = 'bold 16px monospace';
        const label = `${(bbox.confidence * 100).toFixed(0)}%`;
        const labelX = canvasCorners[0].x;
        const labelY = canvasCorners[0].y - 10;
        this.ctx.strokeText(label, labelX, labelY);
        this.ctx.fillText(label, labelX, labelY);

        // Draw handles if selected
        if (drawHandles) {
            // Corner handles
            this.ctx.fillStyle = this.colors.handle;
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 1;
            canvasCorners.forEach(corner => {
                this.ctx.beginPath();
                this.ctx.arc(corner.x, corner.y, this.handleSize, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.stroke();
            });

            // Rotation handle (circle above center)
            const rotHandleX = cx;
            const rotHandleY = cy - this.rotationHandleDistance;
            
            // Line from center to rotation handle
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(cx, cy);
            this.ctx.lineTo(rotHandleX, rotHandleY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);

            // Rotation handle circle
            this.ctx.fillStyle = this.colors.handle;
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(rotHandleX, rotHandleY, this.handleSize, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();
        }

        this.ctx.restore();
    }

    /**
     * Add a new bounding box at the center
     */
    addBoundingBox() {
        if (!this.trainingExample) return;

        const newBbox = new BoundingBox(0.5, 0.5, 0.2, 0.2);
        this.trainingExample.addBoundingBox(newBbox);
        this.selectedBboxIndex = this.trainingExample.boundingBoxes.length - 1;
        this.render();

        if (this.onBboxChanged) {
            this.onBboxChanged(this.selectedBboxIndex, newBbox);
        }
    }

    /**
     * Delete the selected bounding box
     */
    deleteSelectedBoundingBox() {
        if (this.selectedBboxIndex === -1 || !this.trainingExample) return;

        this.trainingExample.removeBoundingBox(this.selectedBboxIndex);
        this.selectedBboxIndex = -1;
        this.render();
    }

    /**
     * Get the currently selected bounding box
     */
    getSelectedBoundingBox() {
        if (this.selectedBboxIndex === -1 || !this.trainingExample) return null;
        return this.trainingExample.boundingBoxes[this.selectedBboxIndex];
    }

    /**
     * Update canvas size to match container
     */
    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.render();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BoundingBoxEditor };
}

