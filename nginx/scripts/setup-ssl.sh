#!/bin/bash

# SSL Certificate Setup Script for CourtSight Production
# This script sets up SSL certificates using Let's Encrypt

set -e

# Configuration
DOMAINS=("courtsight.id" "www.courtsight.id" "api.courtsight.id")
EMAIL="admin@courtsight.id"
STAGING=0  # Set to 1 for testing with staging environment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
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

# Install Certbot if not present
if ! command -v certbot &> /dev/null; then
    log "Installing Certbot..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Generate DH parameters for perfect forward secrecy
if [ ! -f /etc/ssl/certs/dhparam.pem ]; then
    log "Generating DH parameters (this may take a while)..."
    openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
fi

# Create web root for challenges
mkdir -p /var/www/certbot

# Set Certbot command based on staging/production
if [ $STAGING -eq 1 ]; then
    CERTBOT_CMD="certbot certonly --webroot --staging"
    warn "Using Let's Encrypt staging environment"
else
    CERTBOT_CMD="certbot certonly --webroot"
    log "Using Let's Encrypt production environment"
fi

# Request certificates for each domain
for domain in "${DOMAINS[@]}"; do
    log "Requesting certificate for $domain..."
    
    # Check if certificate already exists
    if [ -d "/etc/letsencrypt/live/$domain" ]; then
        warn "Certificate for $domain already exists. Skipping..."
        continue
    fi
    
    # Request certificate
    $CERTBOT_CMD \
        --webroot-path=/var/www/certbot \
        --email $EMAIL \
        --agree-tos \
        --no-eff-email \
        -d $domain
    
    if [ $? -eq 0 ]; then
        log "Certificate for $domain obtained successfully"
    else
        error "Failed to obtain certificate for $domain"
    fi
done

# Test Nginx configuration
log "Testing Nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    log "Nginx configuration is valid"
    
    # Reload Nginx
    log "Reloading Nginx..."
    systemctl reload nginx
    
    log "SSL certificates setup completed successfully!"
    
    # Set up automatic renewal
    log "Setting up automatic certificate renewal..."
    
    # Create renewal script
    cat > /etc/cron.d/certbot-renew << EOF
# Renew Let's Encrypt certificates twice daily
SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

0 */12 * * * root certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF
    
    log "Automatic renewal configured (checks twice daily)"
    
    # Display certificate information
    log "Certificate information:"
    for domain in "${DOMAINS[@]}"; do
        if [ -d "/etc/letsencrypt/live/$domain" ]; then
            echo "Domain: $domain"
            certbot certificates -d $domain | grep -E "(Certificate Name|Domains|Expiry Date)"
            echo ""
        fi
    done
    
else
    error "Nginx configuration test failed. Please check your configuration."
fi

# Security recommendations
log "Security recommendations:"
echo "1. Ensure firewall is configured to allow only necessary ports (80, 443, 22)"
echo "2. Regularly update your system and Nginx"
echo "3. Monitor SSL certificate expiration"
echo "4. Consider implementing fail2ban for additional security"
echo "5. Regular security audits and penetration testing"

log "SSL setup script completed!"