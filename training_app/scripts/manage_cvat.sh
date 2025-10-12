#!/bin/bash

# CVAT Management Script for Trading Card Annotation Project
# This script provides easy commands to manage the CVAT annotation platform

set -e

CVAT_DIR="/home/alec/git/pokemon/cvat_repo"
PROJECT_DIR="/home/alec/git/pokemon"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if CVAT is running
check_cvat_status() {
    cd "$CVAT_DIR"
    if docker compose ps | grep -q "Up"; then
        return 0
    else
        return 1
    fi
}

# Function to start CVAT
start_cvat() {
    print_status "Starting CVAT services..."
    cd "$CVAT_DIR"
    
    # Start core services first
    docker compose up -d cvat_db cvat_redis_inmem cvat_redis_ondisk cvat_clickhouse cvat_opa
    sleep 10
    
    # Start server and UI
    docker compose up -d cvat_server cvat_ui traefik
    sleep 15
    
    if check_cvat_status; then
        print_success "CVAT is now running!"
        print_status "Access CVAT at: http://localhost:8080"
        print_status "Login with username: admin, password: admin123"
    else
        print_error "Failed to start CVAT services"
        return 1
    fi
}

# Function to stop CVAT
stop_cvat() {
    print_status "Stopping CVAT services..."
    cd "$CVAT_DIR"
    docker compose down
    print_success "CVAT services stopped"
}

# Function to restart CVAT
restart_cvat() {
    print_status "Restarting CVAT..."
    stop_cvat
    sleep 5
    start_cvat
}

# Function to show CVAT status
status_cvat() {
    cd "$CVAT_DIR"
    print_status "CVAT Service Status:"
    docker compose ps
    
    if check_cvat_status; then
        print_success "CVAT is running - Access at: http://localhost:8080"
    else
        print_warning "CVAT is not running"
    fi
}

# Function to show CVAT logs
logs_cvat() {
    cd "$CVAT_DIR"
    print_status "Showing CVAT logs (press Ctrl+C to exit)..."
    docker compose logs -f cvat_server cvat_ui
}

# Function to create a backup
backup_cvat() {
    print_status "Creating CVAT backup..."
    cd "$CVAT_DIR"
    
    BACKUP_DIR="$PROJECT_DIR/backups/cvat_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    docker compose exec -T cvat_db pg_dump -U root cvat > "$BACKUP_DIR/cvat_db.sql"
    
    # Backup volumes
    docker run --rm -v cvat_cvat_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/cvat_data.tar.gz -C /data .
    docker run --rm -v cvat_cvat_keys:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/cvat_keys.tar.gz -C /data .
    
    print_success "Backup created at: $BACKUP_DIR"
}

# Function to prepare sample images
prepare_samples() {
    print_status "Preparing sample images for annotation..."
    
    SAMPLE_DIR="$PROJECT_DIR/data/sample_images"
    mkdir -p "$SAMPLE_DIR"
    
    # Copy sample images if they don't exist
    if [ ! "$(ls -A $SAMPLE_DIR)" ]; then
        if [ -d "$PROJECT_DIR/data/occlusion_test/images" ]; then
            cp "$PROJECT_DIR/data/occlusion_test/images"/*.jpg "$SAMPLE_DIR/" 2>/dev/null || true
            print_success "Sample images copied to $SAMPLE_DIR"
        else
            print_warning "No source images found. Please add images to $SAMPLE_DIR manually."
        fi
    else
        print_status "Sample images already exist in $SAMPLE_DIR"
    fi
    
    # List available images
    IMAGE_COUNT=$(ls -1 "$SAMPLE_DIR"/*.jpg 2>/dev/null | wc -l || echo "0")
    print_status "Available images for annotation: $IMAGE_COUNT"
}

# Function to show help
show_help() {
    echo "CVAT Management Script for Trading Card Annotation"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start      Start CVAT services"
    echo "  stop       Stop CVAT services"
    echo "  restart    Restart CVAT services"
    echo "  status     Show CVAT service status"
    echo "  logs       Show CVAT logs (follow mode)"
    echo "  backup     Create a backup of CVAT data"
    echo "  samples    Prepare sample images for annotation"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start CVAT"
    echo "  $0 status    # Check if CVAT is running"
    echo "  $0 samples   # Prepare sample images"
    echo ""
    echo "Access CVAT at: http://localhost:8080"
    echo "Default login: admin / admin123"
}

# Main script logic
case "${1:-help}" in
    start)
        start_cvat
        ;;
    stop)
        stop_cvat
        ;;
    restart)
        restart_cvat
        ;;
    status)
        status_cvat
        ;;
    logs)
        logs_cvat
        ;;
    backup)
        backup_cvat
        ;;
    samples)
        prepare_samples
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
