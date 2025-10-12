/**
 * Dataset Viewer UI Component
 * 
 * Provides a comprehensive interface for viewing, managing, and editing
 * a collection of training examples for computer vision datasets.
 */

class DatasetViewer {
    constructor(containerId, datasetManager) {
        this.container = document.getElementById(containerId);
        this.datasetManager = datasetManager;
        this.editor = null;
        this.currentExampleId = null;
        this.savedScrollPosition = 0;
        
        this.render();
    }

    /**
     * Render the complete UI
     */
    render() {
        // Save scroll position of examples list before re-rendering
        const examplesList = document.getElementById('examplesList');
        if (examplesList) {
            this.savedScrollPosition = examplesList.scrollTop;
        }
        
        // Clean up old editor before destroying DOM
        this.cleanupEditor();

        this.container.innerHTML = `
            <div class="dataset-viewer">
                <div class="dataset-header">
                    <h2>Training Dataset Manager</h2>
                    <div class="dataset-stats" id="datasetStats">
                        ${this.renderStats()}
                    </div>
                </div>

                <div class="dataset-controls">
                    <button id="exportDatasetBtn" class="btn btn-primary">
                        Export Dataset
                    </button>
                    <button id="clearDatasetBtn" class="btn btn-danger">
                        Clear All
                    </button>
                    <button id="saveToStorageBtn" class="btn btn-secondary">
                        Save to Storage
                    </button>
                </div>

                <div class="dataset-content">
                    <div class="examples-list" id="examplesList">
                        ${this.renderExamplesList()}
                    </div>
                    
                    <div class="example-editor" id="exampleEditor">
                        ${this.renderEditor()}
                    </div>
                </div>
            </div>
        `;

        this.attachEventListeners();
        this.initializeEditor();
        
        // Restore scroll position after rendering
        this.restoreScrollPosition();
    }

    /**
     * Restore scroll position of examples list
     */
    restoreScrollPosition() {
        // Use requestAnimationFrame to ensure DOM has been rendered
        requestAnimationFrame(() => {
            const examplesList = document.getElementById('examplesList');
            if (examplesList && this.savedScrollPosition > 0) {
                examplesList.scrollTop = this.savedScrollPosition;
            }
        });
    }

    /**
     * Render dataset statistics
     */
    renderStats() {
        const stats = this.datasetManager.getStats();
        
        // Get storage stats asynchronously
        this.datasetManager.getStorageStats().then(storageStats => {
            const storageTypeEl = document.getElementById('storageType');
            const storageSizeEl = document.getElementById('storageSize');
            if (storageTypeEl) {
                storageTypeEl.textContent = storageStats.storageType;
            }
            if (storageSizeEl) {
                storageSizeEl.textContent = `${storageStats.estimatedSizeMB} MB`;
            }
        }).catch(err => {
            console.error('Failed to get storage stats:', err);
        });
        
        return `
            <div class="stat-item">
                <span class="stat-label">Examples:</span>
                <span class="stat-value">${stats.totalExamples}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Annotations:</span>
                <span class="stat-value">${stats.totalAnnotations}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Avg per Image:</span>
                <span class="stat-value">${stats.avgAnnotationsPerExample}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Storage Type:</span>
                <span class="stat-value" id="storageType">...</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Storage Used:</span>
                <span class="stat-value" id="storageSize">...</span>
            </div>
        `;
    }

    /**
     * Render examples list
     */
    renderExamplesList() {
        const examples = this.datasetManager.getAllExamples();
        
        if (examples.length === 0) {
            return `
                <div class="empty-state">
                    <p>No training examples yet.</p>
                    <p>Use the camera to capture examples with detections.</p>
                </div>
            `;
        }

        return `
            <div class="examples-grid">
                ${examples.map((example, index) => this.renderExampleCard(example, index)).join('')}
            </div>
        `;
    }

    /**
     * Render a single example card
     */
    renderExampleCard(example, index) {
        const summary = example.getSummary();
        const isSelected = example.id === this.currentExampleId;
        
        return `
            <div class="example-card ${isSelected ? 'selected' : ''}" 
                 data-example-id="${example.id}"
                 data-example-index="${index}">
                <div class="example-thumbnail">
                    <img src="${example.imageData}" 
                         alt="Example ${index + 1}"
                         loading="lazy"
                         decoding="async">
                    <div class="example-overlay">
                        <span class="bbox-count">${example.boundingBoxes.length} boxes</span>
                    </div>
                </div>
                <div class="example-info">
                    <div class="example-title">Example ${index + 1}</div>
                    <div class="example-details">
                        <div>${summary.dimensions}</div>
                        <div>${summary.avgConfidence.toFixed(2)} avg conf</div>
                        <div class="example-time">${this.formatTime(summary.timestamp)}</div>
                    </div>
                </div>
                <div class="example-actions">
                    <button class="btn-icon edit-btn" data-action="edit" title="Edit">
                        <span>✏️</span>
                    </button>
                    <button class="btn-icon delete-btn" data-action="delete" title="Delete">
                        <span>🗑️</span>
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render the editor panel
     */
    renderEditor() {
        if (!this.currentExampleId) {
            return `
                <div class="editor-empty">
                    <p>Select an example to edit</p>
                </div>
            `;
        }

        const example = this.datasetManager.getExample(this.currentExampleId);
        if (!example) {
            return `<div class="editor-empty"><p>Example not found</p></div>`;
        }

        return `
            <div class="editor-container">
                <div class="editor-header">
                    <h3>Edit Example</h3>
                    <button id="closeEditorBtn" class="btn btn-sm">Close</button>
                </div>
                
                <div class="editor-canvas-container">
                    <img id="editorImage" src="${example.imageData}" style="display: none;">
                    <canvas id="editorCanvas" width="640" height="480"></canvas>
                </div>

                <div class="editor-toolbar">
                    <button id="addBboxBtn" class="btn btn-sm">Add Box</button>
                    <button id="deleteBboxBtn" class="btn btn-sm btn-danger" disabled>Delete Selected</button>
                    <button id="duplicateBboxBtn" class="btn btn-sm" disabled>Duplicate</button>
                </div>

                <div class="bbox-list" id="bboxList">
                    ${this.renderBboxList(example)}
                </div>

                <div class="bbox-properties" id="bboxProperties">
                    ${this.renderBboxProperties()}
                </div>
            </div>
        `;
    }

    /**
     * Render bounding box list
     */
    renderBboxList(example) {
        if (example.boundingBoxes.length === 0) {
            return '<div class="empty-state-small">No bounding boxes</div>';
        }

        return `
            <div class="bbox-items">
                ${example.boundingBoxes.map((bbox, index) => `
                    <div class="bbox-item" data-bbox-index="${index}">
                        <span class="bbox-number">#${index + 1}</span>
                        <span class="bbox-confidence">${(bbox.confidence * 100).toFixed(1)}%</span>
                        <span class="bbox-size">${(bbox.width * 100).toFixed(1)}% × ${(bbox.height * 100).toFixed(1)}%</span>
                        <span class="bbox-rotation">${(bbox.rotation * 180 / Math.PI).toFixed(1)}°</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Render bounding box properties editor
     */
    renderBboxProperties() {
        if (!this.editor || !this.editor.getSelectedBoundingBox()) {
            return '<div class="empty-state-small">No box selected</div>';
        }

        const bbox = this.editor.getSelectedBoundingBox();
        
        return `
            <div class="properties-form">
                <div class="form-group">
                    <label>Position X:</label>
                    <input type="number" id="bboxX" value="${bbox.x.toFixed(4)}" 
                           min="0" max="1" step="0.001">
                </div>
                <div class="form-group">
                    <label>Position Y:</label>
                    <input type="number" id="bboxY" value="${bbox.y.toFixed(4)}" 
                           min="0" max="1" step="0.001">
                </div>
                <div class="form-group">
                    <label>Width:</label>
                    <input type="number" id="bboxWidth" value="${bbox.width.toFixed(4)}" 
                           min="0.01" max="1" step="0.001">
                </div>
                <div class="form-group">
                    <label>Height:</label>
                    <input type="number" id="bboxHeight" value="${bbox.height.toFixed(4)}" 
                           min="0.01" max="1" step="0.001">
                </div>
                <div class="form-group">
                    <label>Rotation (degrees):</label>
                    <input type="number" id="bboxRotation" 
                           value="${(bbox.rotation * 180 / Math.PI).toFixed(2)}" 
                           min="-180" max="180" step="0.1">
                </div>
                <div class="form-group">
                    <label>Confidence:</label>
                    <input type="number" id="bboxConfidence" value="${bbox.confidence.toFixed(3)}" 
                           min="0" max="1" step="0.01">
                </div>
            </div>
        `;
    }

    /**
     * Initialize the bounding box editor
     */
    initializeEditor() {
        if (!this.currentExampleId) {
            console.warn('initializeEditor: No current example ID');
            return;
        }

        const canvas = document.getElementById('editorCanvas');
        const image = document.getElementById('editorImage');
        
        if (!canvas) {
            console.error('initializeEditor: Canvas element not found');
            return;
        }
        
        if (!image) {
            console.error('initializeEditor: Image element not found');
            return;
        }

        // Function to initialize the editor once image is ready
        const initEditor = () => {
            // Set canvas size to match image aspect ratio
            const maxWidth = 640;
            const maxHeight = 480;
            
            if (!image.naturalWidth || !image.naturalHeight) {
                console.error('initializeEditor: Image has no natural dimensions');
                return;
            }
            
            const imgAspect = image.naturalWidth / image.naturalHeight;
            const containerAspect = maxWidth / maxHeight;

            if (imgAspect > containerAspect) {
                canvas.width = maxWidth;
                canvas.height = maxWidth / imgAspect;
            } else {
                canvas.height = maxHeight;
                canvas.width = maxHeight * imgAspect;
            }

            // Also set CSS dimensions to match canvas resolution
            // This prevents the browser from stretching the canvas
            canvas.style.width = canvas.width + 'px';
            canvas.style.height = canvas.height + 'px';

            console.log('Creating BoundingBoxEditor', { 
                canvasWidth: canvas.width, 
                canvasHeight: canvas.height,
                imgAspect: imgAspect.toFixed(3),
                imageNaturalWidth: image.naturalWidth,
                imageNaturalHeight: image.naturalHeight
            });
            
            this.editor = new BoundingBoxEditor(canvas, image);
            
            const example = this.datasetManager.getExample(this.currentExampleId);
            if (!example) {
                console.error('initializeEditor: Example not found', this.currentExampleId);
                return;
            }
            
            this.editor.loadExample(example);
            console.log('Editor loaded with example', { id: example.id, bboxCount: example.boundingBoxes.length });

            // Set up callbacks
            this.editor.onBboxChanged = async (index, bbox) => {
                this.updateBboxList();
                this.updateBboxProperties();
                await this.datasetManager.saveToStorage();
            };

            this.editor.onBboxSelected = (index) => {
                this.updateBboxProperties();
                this.highlightBboxInList(index);
                const deleteBboxBtn = document.getElementById('deleteBboxBtn');
                const duplicateBboxBtn = document.getElementById('duplicateBboxBtn');
                if (deleteBboxBtn) deleteBboxBtn.disabled = false;
                if (duplicateBboxBtn) duplicateBboxBtn.disabled = false;
            };

            this.setupEditorEventListeners();
            console.log('Editor initialization complete');
        };

        // Check if image is already loaded (cached)
        if (image.complete && image.naturalWidth > 0) {
            console.log('initializeEditor: Image already loaded, initializing immediately');
            // Image is already loaded - use setTimeout to ensure any pending operations complete
            setTimeout(() => initEditor(), 0);
        } else {
            console.log('initializeEditor: Waiting for image to load');
            // Wait for image to load
            image.onload = () => {
                console.log('initializeEditor: Image loaded');
                initEditor();
            };
            
            // Handle load errors
            image.onerror = () => {
                console.error('Failed to load editor image');
            };
        }
    }

    /**
     * Setup event listeners for editor controls
     */
    setupEditorEventListeners() {
        const addBboxBtn = document.getElementById('addBboxBtn');
        const deleteBboxBtn = document.getElementById('deleteBboxBtn');
        const duplicateBboxBtn = document.getElementById('duplicateBboxBtn');
        const closeEditorBtn = document.getElementById('closeEditorBtn');

        if (addBboxBtn) {
            addBboxBtn.addEventListener('click', async () => {
                this.editor.addBoundingBox();
                this.updateBboxList();
                await this.datasetManager.saveToStorage();
            });
        }

        if (deleteBboxBtn) {
            deleteBboxBtn.addEventListener('click', async () => {
                if (confirm('Delete this bounding box?')) {
                    this.editor.deleteSelectedBoundingBox();
                    this.updateBboxList();
                    this.updateBboxProperties();
                    await this.datasetManager.saveToStorage();
                    deleteBboxBtn.disabled = true;
                    duplicateBboxBtn.disabled = true;
                }
            });
        }

        if (duplicateBboxBtn) {
            duplicateBboxBtn.addEventListener('click', async () => {
                const selected = this.editor.getSelectedBoundingBox();
                if (selected) {
                    const duplicate = selected.clone();
                    duplicate.x += 0.05;
                    duplicate.y += 0.05;
                    const example = this.datasetManager.getExample(this.currentExampleId);
                    example.addBoundingBox(duplicate);
                    this.editor.render();
                    this.updateBboxList();
                    await this.datasetManager.saveToStorage();
                }
            });
        }

        if (closeEditorBtn) {
            closeEditorBtn.addEventListener('click', () => {
                this.currentExampleId = null;
                this.cleanupEditor();
                
                // Update selection state in list
                document.querySelectorAll('.example-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                // Update editor panel to show empty state
                this.updateEditorPanel();
            });
        }

        // Setup click handlers for bbox list items
        const bboxItems = document.querySelectorAll('.bbox-item');
        bboxItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                if (this.editor) {
                    this.editor.selectBoundingBox(index);
                    this.highlightBboxInList(index);
                    this.updateBboxProperties();
                }
            });
        });

        // Setup property input listeners
        this.setupPropertyInputListeners();
    }

    /**
     * Setup listeners for property inputs
     */
    setupPropertyInputListeners() {
        const inputs = ['bboxX', 'bboxY', 'bboxWidth', 'bboxHeight', 'bboxRotation', 'bboxConfidence'];
        
        inputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                input.addEventListener('change', () => {
                    this.updateBboxFromProperties();
                });
            }
        });
    }

    /**
     * Update bounding box from property inputs
     */
    async updateBboxFromProperties() {
        const bbox = this.editor.getSelectedBoundingBox();
        if (!bbox) return;

        const x = parseFloat(document.getElementById('bboxX').value);
        const y = parseFloat(document.getElementById('bboxY').value);
        const width = parseFloat(document.getElementById('bboxWidth').value);
        const height = parseFloat(document.getElementById('bboxHeight').value);
        const rotation = parseFloat(document.getElementById('bboxRotation').value) * Math.PI / 180;
        const confidence = parseFloat(document.getElementById('bboxConfidence').value);

        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
        bbox.rotation = rotation;
        bbox.confidence = confidence;

        this.editor.render();
        this.updateBboxList();
        await this.datasetManager.saveToStorage();
    }

    /**
     * Update bounding box list display
     */
    updateBboxList() {
        const bboxListContainer = document.getElementById('bboxList');
        if (!bboxListContainer) return;

        const example = this.datasetManager.getExample(this.currentExampleId);
        if (!example) return;

        bboxListContainer.innerHTML = this.renderBboxList(example);
    }

    /**
     * Update bounding box properties display
     */
    updateBboxProperties() {
        const propertiesContainer = document.getElementById('bboxProperties');
        if (!propertiesContainer) return;

        propertiesContainer.innerHTML = this.renderBboxProperties();
        this.setupPropertyInputListeners();
    }

    /**
     * Highlight a bounding box in the list
     */
    highlightBboxInList(index) {
        const bboxItems = document.querySelectorAll('.bbox-item');
        bboxItems.forEach((item, i) => {
            if (i === index) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Export dataset button
        const exportBtn = document.getElementById('exportDatasetBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportDataset());
        }

        // Clear dataset button
        const clearBtn = document.getElementById('clearDatasetBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearDataset());
        }

        // Save to storage button
        const saveBtn = document.getElementById('saveToStorageBtn');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                await this.datasetManager.saveToStorage();
                const stats = await this.datasetManager.getStorageStats();
                alert(`Dataset saved to ${stats.storageType}!\n${stats.exampleCount} examples, ${stats.estimatedSizeMB} MB`);
            });
        }

        // Example card clicks
        this.container.addEventListener('click', (e) => {
            const card = e.target.closest('.example-card');
            if (card) {
                const action = e.target.closest('[data-action]')?.dataset.action;
                const exampleId = card.dataset.exampleId;

                if (action === 'delete') {
                    this.deleteExample(exampleId);
                } else if (action === 'edit' || !action) {
                    this.editExample(exampleId);
                }
            }
        });
    }

    /**
     * Edit an example
     */
    editExample(exampleId) {
        this.currentExampleId = exampleId;
        
        // Update selection state in list without full re-render
        document.querySelectorAll('.example-card').forEach(card => {
            if (card.dataset.exampleId === exampleId) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
        
        // Update only the editor panel
        this.updateEditorPanel();
    }

    /**
     * Update only the editor panel without re-rendering the examples list
     */
    updateEditorPanel() {
        const editorContainer = document.getElementById('exampleEditor');
        if (!editorContainer) return;
        
        // Clean up old editor first
        this.cleanupEditor();
        
        // Update editor HTML
        editorContainer.innerHTML = this.renderEditor();
        
        // Initialize new editor after DOM has updated
        // Use requestAnimationFrame to ensure DOM is ready
        requestAnimationFrame(() => {
            this.initializeEditor();
        });
    }

    /**
     * Clean up the current editor and its resources
     */
    cleanupEditor() {
        if (this.editor) {
            // Call destroy method to remove event listeners properly
            if (typeof this.editor.destroy === 'function') {
                this.editor.destroy();
            }
            
            // Clear editor reference
            this.editor = null;
        }
        
        // Remove image onload handler
        const image = document.getElementById('editorImage');
        if (image) {
            image.onload = null;
        }
    }

    /**
     * Delete an example
     */
    async deleteExample(exampleId) {
        if (confirm('Delete this training example?')) {
            await this.datasetManager.removeExample(exampleId);
            if (this.currentExampleId === exampleId) {
                this.currentExampleId = null;
            }
            this.render();
        }
    }

    /**
     * Clear entire dataset
     */
    async clearDataset() {
        if (confirm('Clear ALL training examples? This cannot be undone!')) {
            await this.datasetManager.clearAll();
            this.currentExampleId = null;
            this.render();
        }
    }

    /**
     * Export dataset
     */
    async exportDataset() {
        const exports = await this.datasetManager.exportDataset('yolo');
        
        if (exports.length === 0) {
            alert('No examples to export!');
            return;
        }

        // Download each file
        for (const file of exports) {
            if (file.type === 'image') {
                // Download image
                const a = document.createElement('a');
                a.href = file.data;
                a.download = file.filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } else if (file.type === 'text') {
                // Download text file
                const blob = new Blob([file.data], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = file.filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        }

        alert(`Exported ${exports.length / 2} training examples!`);
    }

    /**
     * Format timestamp
     */
    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    }

    /**
     * Refresh the viewer
     */
    refresh() {
        this.render();
    }

    /**
     * Add a new example to the dataset
     */
    async addExample(example) {
        await this.datasetManager.addExample(example);
        this.refresh();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DatasetViewer };
}

