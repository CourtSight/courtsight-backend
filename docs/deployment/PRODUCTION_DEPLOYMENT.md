# CourtSight Production Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying CourtSight API to production with Nginx, SSL certificates, and security best practices.

## Prerequisites

### System Requirements
- Ubuntu 20.04 LTS or later (recommended)
- Minimum 4GB RAM, 2 CPU cores
- 50GB storage space
- Domain name pointed to your server
- Root or sudo access

### Software Requirements
- Docker Engine 20.10+
- Docker Compose v2.0+
- Git
- Nginx (will be installed by script)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd courtsight
   ```

2. **Configure environment**
   ```bash
   cp .env.production.example .env.production
   # Edit .env.production with your actual values
   nano .env.production
   ```

3. **Deploy Nginx configuration**
   ```bash
   chmod +x nginx/scripts/deploy.sh
   sudo ./nginx/scripts/deploy.sh
   ```

4. **Setup SSL certificates**
   ```bash
   chmod +x nginx/scripts/setup-ssl.sh
   sudo ./nginx/scripts/setup-ssl.sh
   ```

5. **Start the application**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Detailed Configuration

### 1. Environment Variables

Create `.env.production` from the example file and configure:

#### Security Settings
```bash
SECRET_KEY=generate-a-strong-secret-key-here
DB_PASSWORD=create-strong-database-password
REDIS_PASSWORD=create-strong-redis-password
```

#### Domain Configuration
```bash
ALLOWED_HOSTS=["your-domain.com", "www.your-domain.com", "api.your-domain.com"]
CORS_ORIGINS=["https://your-domain.com", "https://www.your-domain.com"]
```

### 2. Domain Setup

#### DNS Configuration
Point your domains to your server IP:
```
A    your-domain.com        -> YOUR_SERVER_IP
A    www.your-domain.com    -> YOUR_SERVER_IP
A    api.your-domain.com    -> YOUR_SERVER_IP
```

#### SSL Certificate Setup
The setup script will automatically request certificates for:
- `your-domain.com`
- `www.your-domain.com`
- `api.your-domain.com`

### 3. Nginx Configuration

#### Main Features
- **SSL/TLS**: Modern TLS 1.2/1.3 with strong ciphers
- **Security Headers**: HSTS, CSP, X-Frame-Options, etc.
- **Rate Limiting**: API endpoint protection
- **Caching**: Static files and API response caching
- **Compression**: Gzip and Brotli compression
- **Load Balancing**: Ready for multiple backend instances

#### Key Configuration Files
- `nginx/nginx.conf` - Main Nginx configuration
- `nginx/sites-available/courtsight-api` - Site-specific configuration
- `nginx/conf.d/security.conf` - Security settings
- `nginx/conf.d/ssl.conf` - SSL/TLS optimization
- `nginx/conf.d/cache.conf` - Caching configuration
- `nginx/conf.d/ratelimit.conf` - Rate limiting rules

### 4. Security Features

#### HTTP Security Headers
- Strict-Transport-Security (HSTS)
- Content-Security-Policy (CSP)
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection
- Referrer-Policy

#### Rate Limiting
- General API: 30 requests/minute
- Authentication: 5 requests/minute
- Chatbot: 10 requests/minute
- File uploads: 3 requests/minute
- Admin: 10 requests/hour

#### SSL Configuration
- TLS 1.2/1.3 only
- Strong cipher suites
- Perfect Forward Secrecy
- OCSP stapling
- Session resumption

### 5. Performance Optimization

#### Caching Strategy
- Static files: 1 year cache
- API responses: 5-10 minutes
- Browser caching with ETags
- Redis for application cache

#### Compression
- Gzip for text files
- Brotli support (if available)
- Image optimization

#### Connection Optimization
- HTTP/2 support
- Keep-alive connections
- Connection pooling

## Deployment Steps

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. Application Deployment

```bash
# Clone repository
git clone <your-repo-url>
cd courtsight

# Configure environment
cp .env.production.example .env.production
nano .env.production

# Deploy Nginx
sudo ./nginx/scripts/deploy.sh

# Setup SSL
sudo ./nginx/scripts/setup-ssl.sh

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Verification

```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Test endpoints
curl -I https://api.your-domain.com/health
curl -I https://your-domain.com

# SSL test
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

## Monitoring and Maintenance

### Health Checks

All services include health checks:
- API: `/api/health` endpoint
- Database: PostgreSQL ready check
- Redis: Connection test
- Nginx: Configuration validation

### Log Management

Logs are stored in:
- Nginx: `/var/log/nginx/`
- Application: `./logs/`
- Docker: `docker-compose logs`

### Backup Strategy

Recommended backup schedule:
- Database: Daily automated backups
- Application files: Weekly backups
- SSL certificates: Included in system backup

### Updates

```bash
# Application updates
git pull origin main
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# System updates
sudo apt update && sudo apt upgrade -y
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Issues**
   ```bash
   # Check certificate status
   sudo certbot certificates
   
   # Renew certificates manually
   sudo certbot renew --dry-run
   ```

2. **Nginx Configuration Errors**
   ```bash
   # Test configuration
   sudo nginx -t
   
   # Reload configuration
   sudo systemctl reload nginx
   ```

3. **Service Connection Issues**
   ```bash
   # Check service logs
   docker-compose -f docker-compose.prod.yml logs service-name
   
   # Restart services
   docker-compose -f docker-compose.prod.yml restart
   ```

### Performance Tuning

1. **Database Optimization**
   - Tune PostgreSQL configuration
   - Monitor query performance
   - Regular VACUUM and ANALYZE

2. **Cache Optimization**
   - Monitor cache hit rates
   - Adjust TTL values
   - Implement cache warming

3. **Load Testing**
   ```bash
   # Test with Apache Bench
   ab -n 1000 -c 10 https://api.your-domain.com/health
   
   # Test with hey
   hey -n 1000 -c 10 https://api.your-domain.com/health
   ```

## Security Checklist

- [ ] Strong passwords and secrets
- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Rate limiting configured
- [ ] Security headers enabled
- [ ] CORS properly configured
- [ ] Database access restricted
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Access logging enabled

## Support

For issues and questions:
1. Check logs first
2. Verify configuration
3. Test individual components
4. Consult documentation
5. Contact support team

## Additional Resources

- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)