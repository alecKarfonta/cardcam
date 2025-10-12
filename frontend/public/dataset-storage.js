/**
 * Dataset Storage System
 * 
 * Provides persistent storage for training datasets using IndexedDB (with localStorage fallback).
 * IndexedDB supports much larger storage quotas (50+ MB typical) compared to localStorage (5-10 MB).
 */

class DatasetStorage {
    constructor(dbName = 'CVTrainingDataset', storeName = 'examples', version = 1) {
        this.dbName = dbName;
        this.storeName = storeName;
        this.version = version;
        this.db = null;
        this.fallbackToLocalStorage = false;
        this.localStorageKey = 'cv_training_dataset';
    }

    /**
     * Initialize the database
     */
    async init() {
        if (!window.indexedDB) {
            console.warn('IndexedDB not available, falling back to localStorage');
            this.fallbackToLocalStorage = true;
            return true;
        }

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => {
                console.error('IndexedDB failed to open, falling back to localStorage:', request.error);
                this.fallbackToLocalStorage = true;
                resolve(true);
            };

            request.onsuccess = () => {
                this.db = request.result;
                console.log('IndexedDB initialized successfully');
                resolve(true);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create object store if it doesn't exist
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const objectStore = db.createObjectStore(this.storeName, { keyPath: 'id' });
                    objectStore.createIndex('timestamp', 'timestamp', { unique: false });
                    console.log('IndexedDB object store created');
                }
            };
        });
    }

    /**
     * Save all examples to storage
     */
    async saveAll(examples) {
        if (this.fallbackToLocalStorage) {
            return this._saveToLocalStorage(examples);
        }

        if (!this.db) {
            await this.init();
        }

        try {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);

            // Clear existing data
            await this._promisifyRequest(store.clear());

            // Add all examples
            for (const example of examples) {
                await this._promisifyRequest(store.add({
                    id: example.id,
                    data: example.toJSON(),
                    timestamp: new Date(example.timestamp).getTime()
                }));
            }

            console.log(`Saved ${examples.length} examples to IndexedDB`);
            return true;
        } catch (error) {
            console.error('Failed to save to IndexedDB:', error);
            
            // Try localStorage as fallback
            console.log('Attempting localStorage fallback...');
            this.fallbackToLocalStorage = true;
            return this._saveToLocalStorage(examples);
        }
    }

    /**
     * Load all examples from storage
     */
    async loadAll() {
        if (this.fallbackToLocalStorage) {
            return this._loadFromLocalStorage();
        }

        if (!this.db) {
            await this.init();
            
            if (this.fallbackToLocalStorage) {
                return this._loadFromLocalStorage();
            }
        }

        try {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAll();

            const results = await this._promisifyRequest(request);
            const examples = results.map(item => item.data);
            
            console.log(`Loaded ${examples.length} examples from IndexedDB`);
            return examples;
        } catch (error) {
            console.error('Failed to load from IndexedDB:', error);
            
            // Try localStorage as fallback
            console.log('Attempting localStorage fallback...');
            this.fallbackToLocalStorage = true;
            return this._loadFromLocalStorage();
        }
    }

    /**
     * Delete a single example by ID
     */
    async deleteExample(exampleId) {
        if (this.fallbackToLocalStorage) {
            const examples = await this._loadFromLocalStorage();
            const filtered = examples.filter(ex => ex.id !== exampleId);
            return this._saveToLocalStorage(filtered.map(ex => ({ toJSON: () => ex })));
        }

        if (!this.db) {
            await this.init();
        }

        try {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            await this._promisifyRequest(store.delete(exampleId));
            console.log(`Deleted example ${exampleId} from IndexedDB`);
            return true;
        } catch (error) {
            console.error('Failed to delete from IndexedDB:', error);
            return false;
        }
    }

    /**
     * Clear all data
     */
    async clearAll() {
        if (this.fallbackToLocalStorage) {
            localStorage.removeItem(this.localStorageKey);
            console.log('Cleared all data from localStorage');
            return true;
        }

        if (!this.db) {
            await this.init();
        }

        try {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            await this._promisifyRequest(store.clear());
            console.log('Cleared all data from IndexedDB');
            return true;
        } catch (error) {
            console.error('Failed to clear IndexedDB:', error);
            return false;
        }
    }

    /**
     * Get storage statistics
     */
    async getStats() {
        const examples = await this.loadAll();
        const storageType = this.fallbackToLocalStorage ? 'localStorage' : 'IndexedDB';
        
        let estimatedSize = 0;
        try {
            const jsonStr = JSON.stringify(examples);
            estimatedSize = jsonStr.length;
        } catch (e) {
            console.error('Failed to estimate storage size:', e);
        }

        return {
            storageType,
            exampleCount: examples.length,
            estimatedSizeMB: (estimatedSize / (1024 * 1024)).toFixed(2),
            estimatedSizeBytes: estimatedSize
        };
    }

    /**
     * Convert IndexedDB request to Promise
     */
    _promisifyRequest(request) {
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * Save to localStorage (fallback)
     */
    _saveToLocalStorage(examples) {
        try {
            const data = {
                version: '1.0',
                examples: examples.map(ex => ex.toJSON()),
                savedAt: new Date().toISOString()
            };
            localStorage.setItem(this.localStorageKey, JSON.stringify(data));
            console.log(`Saved ${examples.length} examples to localStorage`);
            return true;
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
            if (e.name === 'QuotaExceededError') {
                throw new Error('Storage quota exceeded. Please export and clear some examples to free up space.');
            }
            throw e;
        }
    }

    /**
     * Load from localStorage (fallback)
     */
    _loadFromLocalStorage() {
        try {
            const dataStr = localStorage.getItem(this.localStorageKey);
            if (dataStr) {
                const data = JSON.parse(dataStr);
                console.log(`Loaded ${data.examples.length} examples from localStorage`);
                return data.examples;
            }
            return [];
        } catch (e) {
            console.error('Failed to load from localStorage:', e);
            return [];
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DatasetStorage };
}

