-- Trading Card Segmentation Database Schema
-- Initialize database for card metadata and annotations

-- Create database if not exists (handled by docker-compose)
-- CREATE DATABASE IF NOT EXISTS card_segmentation;

-- Card Games and Types
CREATE TABLE IF NOT EXISTS card_games (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    abbreviation VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial card games
INSERT INTO card_games (name, abbreviation) VALUES 
    ('Magic: The Gathering', 'MTG'),
    ('Pokemon Trading Card Game', 'PTCG'),
    ('Yu-Gi-Oh!', 'YGO'),
    ('Baseball Cards', 'MLB'),
    ('Basketball Cards', 'NBA'),
    ('Football Cards', 'NFL')
ON CONFLICT (name) DO NOTHING;

-- Card Sets/Expansions
CREATE TABLE IF NOT EXISTS card_sets (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES card_games(id),
    name VARCHAR(200) NOT NULL,
    code VARCHAR(20),
    release_date DATE,
    total_cards INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual Cards
CREATE TABLE IF NOT EXISTS cards (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES card_games(id),
    set_id INTEGER REFERENCES card_sets(id),
    name VARCHAR(200) NOT NULL,
    card_number VARCHAR(50),
    rarity VARCHAR(50),
    card_type VARCHAR(100),
    mana_cost VARCHAR(50), -- For MTG
    power_toughness VARCHAR(20), -- For MTG creatures
    hp INTEGER, -- For Pokemon
    attack INTEGER, -- For Pokemon/YGO
    defense INTEGER, -- For YGO
    level_rank INTEGER, -- For Pokemon/YGO
    description TEXT,
    artist VARCHAR(200),
    flavor_text TEXT,
    market_price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Card Images (source images from APIs)
CREATE TABLE IF NOT EXISTS card_images (
    id SERIAL PRIMARY KEY,
    card_id INTEGER REFERENCES cards(id),
    image_url VARCHAR(500),
    image_type VARCHAR(50), -- 'normal', 'small', 'large', 'art_crop', etc.
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    format VARCHAR(10), -- 'jpg', 'png', 'webp'
    local_path VARCHAR(500), -- Local storage path
    downloaded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training Images (multi-card scenes for segmentation)
CREATE TABLE IF NOT EXISTS training_images (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_path VARCHAR(500),
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    format VARCHAR(10),
    source VARCHAR(100), -- 'api', 'community', 'synthetic', 'manual'
    scene_type VARCHAR(50), -- 'single_card', 'multi_card', 'binder', 'sheet'
    card_count INTEGER,
    lighting_condition VARCHAR(50), -- 'natural', 'artificial', 'flash', 'mixed'
    background_type VARCHAR(50), -- 'table', 'binder', 'sleeve', 'fabric'
    quality_score DECIMAL(3,2), -- 0.0 to 1.0
    annotation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'reviewed'
    annotator_id VARCHAR(100),
    reviewer_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Annotations for segmentation masks
CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    training_image_id INTEGER REFERENCES training_images(id),
    card_id INTEGER REFERENCES cards(id), -- NULL if card not identified
    instance_id INTEGER, -- For multiple instances of same card
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    segmentation_mask TEXT, -- JSON or compressed polygon data
    confidence_score DECIMAL(3,2),
    annotation_tool VARCHAR(50), -- 'cvat', 'labelstudio', 'manual', 'sam', 'yolo'
    annotation_time INTEGER, -- Seconds spent annotating
    quality_flags TEXT[], -- Array of quality issues
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Training Runs
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(50), -- 'yolo11', 'mask_rcnn', 'sam', 'ensemble'
    config_path VARCHAR(500),
    dataset_version VARCHAR(50),
    training_images_count INTEGER,
    validation_images_count INTEGER,
    test_images_count INTEGER,
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate DECIMAL(10,8),
    optimizer VARCHAR(50),
    loss_function VARCHAR(100),
    best_map_score DECIMAL(5,4),
    best_epoch INTEGER,
    training_time INTEGER, -- Minutes
    gpu_hours DECIMAL(8,2),
    status VARCHAR(20), -- 'running', 'completed', 'failed', 'stopped'
    model_path VARCHAR(500),
    logs_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Model Evaluation Results
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    training_run_id INTEGER REFERENCES training_runs(id),
    dataset_split VARCHAR(20), -- 'train', 'validation', 'test'
    map_50 DECIMAL(5,4), -- mAP@0.5
    map_50_95 DECIMAL(5,4), -- mAP@0.5:0.95
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    inference_time_ms INTEGER,
    gpu_memory_mb INTEGER,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Collection API Logs
CREATE TABLE IF NOT EXISTS api_collection_logs (
    id SERIAL PRIMARY KEY,
    api_source VARCHAR(50), -- 'scryfall', 'pokemon_tcg', 'ygoprodeck'
    endpoint VARCHAR(200),
    request_count INTEGER,
    success_count INTEGER,
    error_count INTEGER,
    rate_limit_hits INTEGER,
    data_collected INTEGER, -- Number of cards/images
    collection_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_cards_game_set ON cards(game_id, set_id);
CREATE INDEX IF NOT EXISTS idx_card_images_card_id ON card_images(card_id);
CREATE INDEX IF NOT EXISTS idx_training_images_status ON training_images(annotation_status);
CREATE INDEX IF NOT EXISTS idx_annotations_training_image ON annotations(training_image_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_run ON evaluation_results(training_run_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_cards_updated_at BEFORE UPDATE ON cards
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_images_updated_at BEFORE UPDATE ON training_images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
