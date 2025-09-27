#!/bin/bash

# CVAT Setup Script for Trading Card Annotation
# This script sets up CVAT for local development and annotation

set -e

echo "🚀 Setting up CVAT for Trading Card Annotation..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ./data/cvat_annotations
mkdir -p ./data/cvat_projects
mkdir -p ./data/sample_images

# Check if sample images exist
if [ ! "$(ls -A ./data/sample_images)" ]; then
    echo "📸 Copying sample images for annotation..."
    # Copy some sample images from the processed data
    if [ -d "./data/processed/card_images" ]; then
        find ./data/processed/card_images -name "*.jpg" -o -name "*.png" | head -20 | while read img; do
            cp "$img" ./data/sample_images/
        done
        echo "✅ Copied 20 sample images to ./data/sample_images/"
    else
        echo "⚠️  No processed images found. You'll need to add images to ./data/sample_images/ manually."
    fi
fi

# Start CVAT services
echo "🐳 Starting CVAT services..."
docker-compose -f docker-compose.cvat.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose -f docker-compose.cvat.yml ps

# Create superuser
echo "👤 Creating CVAT superuser..."
echo "Please enter details for the CVAT admin user:"
docker-compose -f docker-compose.cvat.yml exec cvat_server python3 manage.py createsuperuser

echo "✅ CVAT setup complete!"
echo ""
echo "🌐 Access CVAT at: http://localhost:8080"
echo "📊 Traefik dashboard at: http://localhost:8090"
echo ""
echo "📝 Next steps:"
echo "1. Open http://localhost:8080 in your browser"
echo "2. Login with the superuser credentials you just created"
echo "3. Create a new project for card annotation"
echo "4. Upload images from ./data/sample_images/"
echo "5. Start annotating!"
echo ""
echo "🛑 To stop CVAT: docker-compose -f docker-compose.cvat.yml down"
echo "🔄 To restart CVAT: docker-compose -f docker-compose.cvat.yml up -d"
