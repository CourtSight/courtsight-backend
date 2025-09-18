#!/bin/bash

# Directory Setup Script for CourtSight Production
# This script creates and sets proper permissions for application directories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Get the current directory
CURRENT_DIR=$(pwd)
PROJECT_ROOT="$CURRENT_DIR"

# Function to create directory with proper permissions
create_directory() {
    local dir_path="$1"
    local permissions="$2"
    local owner="$3"
    
    if [ ! -d "$dir_path" ]; then
        log "Creating directory: $dir_path"
        mkdir -p "$dir_path"
    else
        info "Directory already exists: $dir_path"
    fi
    
    # Set permissions
    chmod "$permissions" "$dir_path"
    
    # Set ownership if specified and running as root
    if [ -n "$owner" ] && [ "$EUID" -eq 0 ]; then
        chown -R "$owner" "$dir_path"
        log "Set ownership of $dir_path to $owner"
    fi
    
    log "Set permissions $permissions for $dir_path"
}

# Main setup function
setup_directories() {
    log "Starting directory setup for CourtSight..."
    
    # Application directories
    create_directory "$PROJECT_ROOT/logs" "755" "1000:1000"
    create_directory "$PROJECT_ROOT/uploads" "755" "1000:1000"
    create_directory "$PROJECT_ROOT/temp" "755" "1000:1000"
    create_directory "$PROJECT_ROOT/cache" "755" "1000:1000"
    
    # Nginx directories (if running Nginx on host)
    if [ "$1" == "--nginx-host" ]; then
        log "Setting up host-based Nginx directories..."
        
        if [ "$EUID" -eq 0 ]; then
            create_directory "/var/cache/nginx" "755" "nginx:nginx"
            create_directory "/var/log/nginx" "755" "nginx:nginx"
            create_directory "/var/www/courtsight" "755" "nginx:nginx"
            create_directory "/var/www/certbot" "755" "nginx:nginx"
            create_directory "/usr/share/nginx/html" "755" "nginx:nginx"
        else
            warn "Nginx directories require root privileges. Run with sudo for host-based Nginx setup."
        fi
    fi
    
    # Database directories (if using external database)
    if [ "$1" == "--external-db" ]; then
        log "Setting up external database directories..."
        create_directory "$PROJECT_ROOT/database/backups" "700" "1000:1000"
        create_directory "$PROJECT_ROOT/database/dumps" "700" "1000:1000"
    fi
    
    # SSL certificate directories
    if [ "$1" == "--ssl" ] || [ "$2" == "--ssl" ]; then
        log "Setting up SSL certificate directories..."
        
        if [ "$EUID" -eq 0 ]; then
            create_directory "/etc/letsencrypt" "755" "root:root"
            create_directory "/etc/ssl/certs" "755" "root:root"
        else
            warn "SSL directories require root privileges. Run with sudo for SSL setup."
        fi
    fi
    
    log "Directory setup completed successfully!"
}

# Validation function
validate_setup() {
    log "Validating directory setup..."
    
    local required_dirs=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/temp"
        "$PROJECT_ROOT/cache"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            error "Required directory not found: $dir"
        fi
        
        if [ ! -w "$dir" ]; then
            error "Directory not writable: $dir"
        fi
        
        info "✓ Directory OK: $dir"
    done
    
    log "All directories validated successfully!"
}

# Check Docker setup
check_docker_setup() {
    log "Checking Docker setup..."
    
    if ! command -v docker &> /dev/null; then
        warn "Docker not found. Install Docker before running the application."
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        warn "Docker Compose not found. Install Docker Compose before running the application."
        return 1
    fi
    
    info "Docker setup is ready"
    return 0
}

# Fix ownership for existing directories
fix_permissions() {
    log "Fixing permissions for existing directories..."
    
    # Get current user ID (will be 1000 for the app user in container)
    local user_id=${SUDO_UID:-$(id -u)}
    local group_id=${SUDO_GID:-$(id -g)}
    
    # If running as root, use 1000:1000 for Docker container compatibility
    if [ "$EUID" -eq 0 ]; then
        user_id=1000
        group_id=1000
    fi
    
    local dirs_to_fix=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/temp"
        "$PROJECT_ROOT/cache"
    )
    
    for dir in "${dirs_to_fix[@]}"; do
        if [ -d "$dir" ]; then
            log "Fixing permissions for: $dir"
            chmod -R 755 "$dir"
            
            if [ "$EUID" -eq 0 ]; then
                chown -R "$user_id:$group_id" "$dir"
            fi
        fi
    done
    
    log "Permission fixes completed!"
}

# Help function
show_help() {
    echo "Directory Setup Script for CourtSight"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --nginx-host    Setup directories for host-based Nginx (requires sudo)"
    echo "  --external-db   Setup directories for external database"
    echo "  --ssl          Setup SSL certificate directories (requires sudo)"
    echo "  --fix-perms    Fix permissions for existing directories"
    echo "  --validate     Validate existing directory setup"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Basic setup"
    echo "  sudo $0 --nginx-host --ssl  # Full host-based setup with SSL"
    echo "  $0 --fix-perms             # Fix permissions only"
    echo "  $0 --validate              # Validate setup"
}

# Main execution
main() {
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --validate)
            validate_setup
            exit 0
            ;;
        --fix-perms)
            fix_permissions
            exit 0
            ;;
        *)
            setup_directories "$@"
            validate_setup
            check_docker_setup
            
            echo ""
            log "Setup complete! You can now run:"
            echo "  docker-compose up -d                    # For containerized Nginx"
            echo "  docker-compose -f docker-compose.host-nginx.yml up -d  # For host-based Nginx"
            ;;
    esac
}

# Run main function with all arguments
main "$@"