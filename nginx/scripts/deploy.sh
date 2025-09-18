#!/bin/bash

# Nginx Deployment Script for CourtSight Production
# This script deploys the Nginx configuration safely

set -e

# Configuration
NGINX_CONFIG_DIR="/etc/nginx"
BACKUP_DIR="/etc/nginx/backup-$(date +%Y%m%d-%H%M%S)"
SOURCE_DIR="$(dirname "$0")/.."

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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
fi

# Backup existing configuration
backup_config() {
    log "Creating backup of existing Nginx configuration..."
    mkdir -p "$BACKUP_DIR"
    
    if [ -d "$NGINX_CONFIG_DIR" ]; then
        cp -r "$NGINX_CONFIG_DIR"/* "$BACKUP_DIR"/ 2>/dev/null || true
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Validate source configuration
validate_source() {
    log "Validating source configuration..."
    
    if [ ! -f "$SOURCE_DIR/nginx.conf" ]; then
        error "nginx.conf not found in source directory"
    fi
    
    if [ ! -f "$SOURCE_DIR/sites-available/courtsight-api" ]; then
        error "courtsight-api site configuration not found"
    fi
    
    info "Source configuration files validated"
}

# Install Nginx if not present
install_nginx() {
    if ! command -v nginx &> /dev/null; then
        log "Installing Nginx..."
        apt-get update
        apt-get install -y nginx
    else
        info "Nginx is already installed"
    fi
}

# Deploy configuration files
deploy_config() {
    log "Deploying Nginx configuration..."
    
    # Main nginx.conf
    cp "$SOURCE_DIR/nginx.conf" "$NGINX_CONFIG_DIR/nginx.conf"
    
    # Create directories if they don't exist
    mkdir -p "$NGINX_CONFIG_DIR/sites-available"
    mkdir -p "$NGINX_CONFIG_DIR/sites-enabled"
    mkdir -p "$NGINX_CONFIG_DIR/conf.d"
    
    # Copy site configuration
    cp "$SOURCE_DIR/sites-available/courtsight-api" "$NGINX_CONFIG_DIR/sites-available/"
    
    # Copy additional configurations
    cp "$SOURCE_DIR/conf.d"/*.conf "$NGINX_CONFIG_DIR/conf.d/"
    
    # Enable site
    if [ ! -L "$NGINX_CONFIG_DIR/sites-enabled/courtsight-api" ]; then
        ln -s "$NGINX_CONFIG_DIR/sites-available/courtsight-api" "$NGINX_CONFIG_DIR/sites-enabled/"
        log "Site enabled: courtsight-api"
    fi
    
    # Remove default site if exists
    if [ -L "$NGINX_CONFIG_DIR/sites-enabled/default" ]; then
        rm "$NGINX_CONFIG_DIR/sites-enabled/default"
        log "Default site disabled"
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    # Cache directories
    mkdir -p /var/cache/nginx/{static,api,docs,fastcgi}
    chown -R nginx:nginx /var/cache/nginx/ 2>/dev/null || chown -R www-data:www-data /var/cache/nginx/
    
    # Log directories
    mkdir -p /var/log/nginx
    
    # Web root
    mkdir -p /var/www/courtsight
    mkdir -p /var/www/certbot
    
    # HTML error pages
    mkdir -p /usr/share/nginx/html
    
    info "Directories created successfully"
}

# Create custom error pages
create_error_pages() {
    log "Creating custom error pages..."
    
    # 429 Rate Limit page
    cat > /usr/share/nginx/html/429.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Too Many Requests</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #e74c3c; }
        p { color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>429 - Too Many Requests</h1>
    <p>You have exceeded the rate limit. Please try again later.</p>
</body>
</html>
EOF

    # 503 Service Unavailable page
    cat > /usr/share/nginx/html/503.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #f39c12; }
        p { color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>503 - Service Temporarily Unavailable</h1>
    <p>The service is temporarily unavailable. Please try again in a few minutes.</p>
</body>
</html>
EOF

    info "Custom error pages created"
}

# Test configuration
test_config() {
    log "Testing Nginx configuration..."
    
    if nginx -t; then
        log "Nginx configuration test passed"
        return 0
    else
        error "Nginx configuration test failed"
        return 1
    fi
}

# Start/restart Nginx
restart_nginx() {
    log "Restarting Nginx..."
    
    if systemctl is-active --quiet nginx; then
        systemctl reload nginx
        log "Nginx reloaded successfully"
    else
        systemctl start nginx
        systemctl enable nginx
        log "Nginx started and enabled"
    fi
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        ufw allow 'Nginx Full'
        ufw allow ssh
        info "UFW firewall configured"
    elif command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload
        info "Firewalld configured"
    else
        warn "No recognized firewall found. Please configure manually."
    fi
}

# Set up log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/nginx-courtsight << 'EOF'
/var/log/nginx/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 640 nginx adm
    sharedscripts
    postrotate
        if [ -f /var/run/nginx.pid ]; then
            kill -USR1 `cat /var/run/nginx.pid`
        fi
    endscript
}
EOF

    info "Log rotation configured"
}

# Display deployment summary
deployment_summary() {
    log "Deployment Summary:"
    echo "================================"
    echo "✓ Nginx configuration deployed"
    echo "✓ Site enabled: courtsight-api"
    echo "✓ Cache directories created"
    echo "✓ Custom error pages created"
    echo "✓ Firewall configured"
    echo "✓ Log rotation configured"
    echo ""
    echo "Next Steps:"
    echo "1. Run SSL setup: ./scripts/setup-ssl.sh"
    echo "2. Update DNS records to point to this server"
    echo "3. Test all endpoints"
    echo "4. Monitor logs: tail -f /var/log/nginx/access.log"
    echo ""
    echo "Configuration files backed up to: $BACKUP_DIR"
    echo "================================"
}

# Rollback function
rollback() {
    error_msg="$1"
    warn "Deployment failed: $error_msg"
    warn "Rolling back configuration..."
    
    if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR)" ]; then
        cp -r "$BACKUP_DIR"/* "$NGINX_CONFIG_DIR"/
        nginx -t && systemctl reload nginx
        log "Configuration rolled back successfully"
    else
        error "No backup found. Manual intervention required."
    fi
    
    exit 1
}

# Main deployment process
main() {
    log "Starting Nginx deployment for CourtSight..."
    
    # Trap errors for rollback
    trap 'rollback "Unexpected error occurred"' ERR
    
    validate_source
    backup_config
    install_nginx
    create_directories
    deploy_config
    create_error_pages
    
    if test_config; then
        restart_nginx
        configure_firewall
        setup_log_rotation
        deployment_summary
        log "Deployment completed successfully!"
    else
        rollback "Configuration test failed"
    fi
}

# Run main function
main "$@"