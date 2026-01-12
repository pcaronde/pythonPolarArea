# Production Deployment Guide - Ubuntu Server

Complete guide for deploying the Performance Assessment Application on Ubuntu Server with production-grade security and reliability.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Server Setup](#server-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [MongoDB Setup](#mongodb-setup)
5. [Application Deployment](#application-deployment)
6. [Securing Environment Variables](#securing-environment-variables)
7. [Process Management with PM2](#process-management-with-pm2)
8. [Nginx Reverse Proxy](#nginx-reverse-proxy)
9. [SSL/HTTPS with Let's Encrypt](#ssltls-with-lets-encrypt)
10. [Firewall Configuration](#firewall-configuration)
11. [Security Hardening](#security-hardening)
12. [Monitoring and Logs](#monitoring-and-logs)
13. [Backup Strategy](#backup-strategy)

## Prerequisites

- Ubuntu Server 20.04 LTS or 22.04 LTS
- Root or sudo access
- Domain name pointing to your server IP
- At least 2GB RAM, 20GB disk space

## Server Setup

### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Create Application User

**IMPORTANT:** Never run Node.js applications as root. Create a dedicated user:

```bash
# Create user without login shell for security
sudo useradd -r -s /bin/false perfassess

# Create user with login shell (for deployment)
sudo useradd -m -s /bin/bash perfassess
sudo passwd perfassess

# Add to sudo group (if needed for deployment)
sudo usermod -aG sudo perfassess
```

### 3. Set Up Application Directory

```bash
sudo mkdir -p /var/www/performance-assessment
sudo chown perfassess:perfassess /var/www/performance-assessment
```

## Installing Dependencies

### 1. Install Node.js (v18 LTS)

```bash
# Install NodeSource repository
# DEPRECATED OLD curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
NEW curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash - 

# Install Node.js
sudo apt install -y nodejs

# Verify installation
node --version  # Should show v18.x.x
npm --version
```

### 2. Install MongoDB

#### Option A: Local MongoDB Installation

```bash
# Import MongoDB GPG key
curl -fsSL https://pgp.mongodb.com/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

# Update and install
sudo apt update
sudo apt install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
sudo systemctl status mongod
```

#### Option B: MongoDB Atlas (Cloud - Recommended)

1. Create free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create cluster
3. Whitelist your server IP
4. Get connection string
5. Use in .env file

### 3. Install Nginx

```bash
sudo apt install -y nginx

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Verify
sudo systemctl status nginx
```

### 4. Install PM2 (Process Manager)

```bash
sudo npm install -g pm2

# Verify
pm2 --version
```

### 5. Install Certbot (for SSL)

```bash
sudo apt install -y certbot python3-certbot-nginx
```

## MongoDB Setup

### Secure MongoDB (Local Installation)

```bash
# Connect to MongoDB shell
mongosh

# Create admin user
use admin

db.createUser({
  user: "admin",
  pwd: "STRONG_PASSWORD",
  roles: [{ role: "userAdminAnyDatabase", db: "admin" }]
})

# Create application user
use hr_performance

db.createUser({
  user: "perfassess",
  pwd: "STRONG_PASSWORD",
  roles: [{ role: "readWrite", db: "hr_performance" }]
})

exit
```

### Enable Authentication

```bash
# Edit MongoDB config
sudo vim /etc/mongod.conf

# Add/uncomment these lines:
security:
  authorization: enabled

# Restart MongoDB
sudo systemctl restart mongod
```

### Update Connection String

Your MongoDB URI will be:
```
mongodb://perfassess:STRONG_APP_PASSWORD_HERE@localhost:27017/hr_performance?authSource=hr_performance
```

## Application Deployment

### 1. Clone Repository

```bash
# Switch to application user
su - perfassess

# Navigate to app directory
cd /var/www/performance-assessment

# Clone repository
git clone <your-repository-url> .

# Or upload files via SCP
# scp -r /local/path/* perfassess@your-server:/var/www/performance-assessment/
```

### 2. Install Dependencies

```bash
npm install --production
```

### 3. Build (if needed)

```bash
# If you have any build steps
npm run build
```

## Securing Environment Variables

### 1. Create .env File

**NEVER commit .env to git!**

```bash
cd /var/www/performance-assessment
vim .env
```

Add your production configuration:

```env
# Server Configuration
NODE_ENV=production
PORT=5000
FRONTEND_URL=https://yourdomain.com

# MongoDB Configuration
MONGODB_URI=mongodb://perfassess:YOUR_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance
MONGODB_DB_NAME=hr_performance

# JWT Configuration
JWT_SECRET=<GENERATE_SECURE_64_CHAR_SECRET>
JWT_EXPIRES_IN=24h

# Security
BCRYPT_ROUNDS=12

# Registration Control
ALLOW_REGISTRATION=false
```

### 2. Generate Secure JWT Secret

```bash
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
```

Copy output to JWT_SECRET in .env

### 3. Secure .env File Permissions

**CRITICAL SECURITY STEP:**

```bash
# Set restrictive permissions (owner read/write only)
chmod 600 .env

# Verify owner
chown perfassess:perfassess .env

# Verify permissions
ls -la .env
# Should show: -rw------- 1 perfassess perfassess
```

**Explanation:**
- `600` = Owner can read/write, no one else can access
- Only the perfassess user can read this file
- Root can still access (unavoidable)
- Other users cannot read the file

### 4. Verify .gitignore

```bash
cat .gitignore | grep .env
```

Should show:
```
.env
.env.local
.env.*.local
```

### 5. Environment Variable Security Checklist

- [ ] `.env` file has 600 permissions
- [ ] `.env` is owned by application user
- [ ] `.env` is in `.gitignore`
- [ ] `.env` is NOT in git repository
- [ ] JWT_SECRET is at least 64 random characters
- [ ] MongoDB password is strong (16+ chars, mixed case, numbers, symbols)
- [ ] `NODE_ENV=production` is set
- [ ] `ALLOW_REGISTRATION=false` for production

## Process Management with PM2

### 1. Create PM2 Ecosystem File

```bash
vim ecosystem.config.js
```

Add configuration:

```javascript
module.exports = {
  apps: [{
    name: 'performance-assessment',
    script: './backend/server.js',
    instances: 2,  // Or 'max' for all CPU cores
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 5000
    },
    error_file: '/var/log/performance-assessment/error.log',
    out_file: '/var/log/performance-assessment/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    max_restarts: 10,
    min_uptime: '10s',
    max_memory_restart: '500M',
    watch: false  // Don't use watch in production
  }]
};
```

### 2. Create Log Directory

```bash
sudo mkdir -p /var/log/performance-assessment
sudo chown perfassess:perfassess /var/log/performance-assessment
```

### 3. Start Application with PM2

```bash
# Start application
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Generate startup script (run as perfassess user)
pm2 startup

# Copy and run the command it outputs (as root)
# Example: sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u perfassess --hp /home/perfassess
```

### 4. PM2 Commands

```bash
# View status
pm2 status

# View logs
pm2 logs performance-assessment

# Restart application
pm2 restart performance-assessment

# Stop application
pm2 stop performance-assessment

# Monitor
pm2 monit

# Show detailed info
pm2 show performance-assessment
```

## Nginx Reverse Proxy

### 1. Create Nginx Configuration

```bash
sudo vim /etc/nginx/sites-available/performance-assessment
```

Add configuration:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name yourdomain.com www.yourdomain.com;

    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL certificates (will be added by Certbot)
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Root directory for static files
    root /var/www/performance-assessment;
    index index-std.html index-3t.html login.html;

    # Serve static files
    location / {
        try_files $uri $uri/ =404;
    }

    # Proxy API requests to Node.js backend
    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Logging
    access_log /var/log/nginx/performance-assessment-access.log;
    error_log /var/log/nginx/performance-assessment-error.log;

    # Client max body size (for CSV uploads)
    client_max_body_size 10M;
}
```

### 2. Enable Configuration

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/performance-assessment /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### 3. Update Frontend API URL

Edit `frontend/api/apiClient.js`:

```javascript
this.baseURL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : 'https://yourdomain.com/api';  // Update this line
```

## SSL/TLS with Let's Encrypt

### 1. Obtain SSL Certificate

```bash
# Create directory for Let's Encrypt challenges
sudo mkdir -p /var/www/certbot

# Get certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Follow prompts:
# - Enter email address
# - Agree to terms
# - Choose whether to redirect HTTP to HTTPS (Yes)
```

### 2. Auto-Renewal

Certbot automatically sets up renewal. Verify:

```bash
# Test renewal
sudo certbot renew --dry-run

# Check renewal timer
sudo systemctl status certbot.timer
```

### 3. Update .env

```bash
vim .env
```

Update FRONTEND_URL:
```env
FRONTEND_URL=https://yourdomain.com
```

Restart application:
```bash
pm2 restart performance-assessment
```

## Firewall Configuration

### 1. Install UFW (if not installed)

```bash
sudo apt install -y ufw
```

### 2. Configure Firewall Rules

```bash
# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH (IMPORTANT: Don't lock yourself out!)
sudo ufw allow 22/tcp comment 'SSH'

# HTTP and HTTPS
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# MongoDB (only if local, and only from localhost)
# sudo ufw allow from 127.0.0.1 to any port 27017 comment 'MongoDB local'

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### 3. Verify Rules

```bash
sudo ufw status numbered
```

Should show:
```
Status: active

     To                         Action      From
     --                         ------      ----
[ 1] 22/tcp                     ALLOW IN    Anywhere
[ 2] 80/tcp                     ALLOW IN    Anywhere
[ 3] 443/tcp                    ALLOW IN    Anywhere
```

## Security Hardening

### 1. Disable Direct Node.js Access

Ensure Node.js only listens on localhost:

```javascript
// backend/server.js
app.listen(PORT, 'localhost', () => {  // Add 'localhost' here
  console.log(`Server running on port ${PORT}`);
});
```

### 2. Set Up Fail2Ban

Protect against brute force attacks:

```bash
sudo apt install -y fail2ban

# Create custom config
sudo nano /etc/fail2ban/jail.local
```

Add:
```ini
[nginx-http-auth]
enabled = true

[nginx-noscript]
enabled = true

[nginx-badbots]
enabled = true

[nginx-noproxy]
enabled = true
```

Restart:
```bash
sudo systemctl restart fail2ban
sudo systemctl enable fail2ban
```

### 3. Regular Updates

```bash
# Create update script
sudo vim /usr/local/bin/security-updates.sh
```

Add:
```bash
#!/bin/bash
apt update && apt upgrade -y
apt autoremove -y
```

Make executable:
```bash
sudo chmod +x /usr/local/bin/security-updates.sh
```

Set up weekly cron:
```bash
sudo crontab -e

# Add line:
0 2 * * 0 /usr/local/bin/security-updates.sh
```

### 4. Secure MongoDB

- Enable authentication (done earlier)
- Bind to localhost only (edit `/etc/mongod.conf`)
- Keep MongoDB updated
- Regular backups

### 5. Application Security Checklist

- [ ] `NODE_ENV=production` in .env
- [ ] `ALLOW_REGISTRATION=false` in .env
- [ ] Strong JWT secret (64+ characters)
- [ ] HTTPS enabled
- [ ] HSTS header enabled
- [ ] MongoDB authentication enabled
- [ ] Firewall configured (UFW)
- [ ] Fail2Ban installed
- [ ] Node.js binds to localhost only
- [ ] .env file has 600 permissions
- [ ] Application runs as non-root user
- [ ] Regular security updates enabled

## Monitoring and Logs

### 1. View Application Logs

```bash
# PM2 logs
pm2 logs performance-assessment

# Nginx access logs
sudo tail -f /var/log/nginx/performance-assessment-access.log

# Nginx error logs
sudo tail -f /var/log/nginx/performance-assessment-error.log

# MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log

# System logs
sudo journalctl -u mongod -f
```

### 2. Set Up Log Rotation

PM2 handles its own log rotation. For Nginx:

```bash
# Nginx logrotate is usually pre-configured
cat /etc/logrotate.d/nginx
```

### 3. Monitor System Resources

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Use PM2 monitoring
pm2 monit
```

### 4. Set Up Alerts (Optional)

Consider services like:
- [PM2 Plus](https://pm2.io/) (free tier available)
- [Netdata](https://www.netdata.cloud/)
- [Prometheus + Grafana](https://prometheus.io/)

## Backup Strategy

### 1. MongoDB Backups

Create backup script:

```bash
mkdir -p ~/scripts
vim ~/scripts/backup-mongodb.sh
```

Add:
```bash
#!/bin/bash
BACKUP_DIR="/var/backups/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

mongodump \
  --uri="mongodb://perfassess:PASSWORD@localhost:27017/hr_performance?authSource=hr_performance" \
  --out="$BACKUP_DIR/backup_$DATE"

# Keep only last 7 days
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: backup_$DATE"
```

Make executable:
```bash
chmod +x ~/scripts/backup-mongodb.sh
```

Schedule daily backups:
```bash
crontab -e

# Add line (daily at 2 AM):
0 2 * * * /home/perfassess/scripts/backup-mongodb.sh >> /var/log/mongodb-backup.log 2>&1
```

### 2. Application Backups

```bash
# Backup application code
tar -czf /var/backups/app-$(date +%Y%m%d).tar.gz /var/www/performance-assessment

# Backup .env file securely
sudo cp /var/www/performance-assessment/.env /root/.env.backup
sudo chmod 600 /root/.env.backup
```

### 3. Offsite Backups

Consider using:
- AWS S3
- Backblaze B2
- Rsync to another server

Example rsync:
```bash
rsync -avz /var/backups/mongodb/ user@backup-server:/backups/performance-assessment/
```

## Deployment Checklist

Before going live, verify:

### Prerequisites
- [ ] Domain DNS points to server IP
- [ ] Server has at least 2GB RAM
- [ ] Ubuntu 20.04 or 22.04 installed

### Installation
- [ ] Node.js v18 installed
- [ ] MongoDB installed and configured
- [ ] Nginx installed
- [ ] PM2 installed globally
- [ ] Certbot installed

### Application
- [ ] Code deployed to `/var/www/performance-assessment`
- [ ] Dependencies installed (`npm install --production`)
- [ ] `.env` file created with production values
- [ ] `.env` has 600 permissions
- [ ] Application user created (perfassess)
- [ ] Application runs as non-root user

### Database
- [ ] MongoDB authentication enabled
- [ ] Application database user created
- [ ] Connection tested

### Process Management
- [ ] PM2 ecosystem configured
- [ ] Application starts with PM2
- [ ] PM2 startup script enabled
- [ ] Logs directory created

### Web Server
- [ ] Nginx configuration created
- [ ] Configuration syntax tested (`nginx -t`)
- [ ] Nginx reloaded
- [ ] Static files accessible

### SSL/HTTPS
- [ ] SSL certificate obtained
- [ ] HTTP redirects to HTTPS
- [ ] Certificate auto-renewal tested

### Security
- [ ] Firewall enabled (UFW)
- [ ] Only ports 22, 80, 443 open
- [ ] Fail2Ban installed
- [ ] MongoDB bound to localhost
- [ ] Node.js bound to localhost
- [ ] Security headers configured in Nginx

### Monitoring
- [ ] Can view PM2 logs
- [ ] Can view Nginx logs
- [ ] Can view MongoDB logs
- [ ] Backup script created and scheduled

### Testing
- [ ] Can access https://yourdomain.com
- [ ] Can register/login
- [ ] Can create assessment
- [ ] Assessment saves to MongoDB
- [ ] Can view history
- [ ] Can edit/delete assessments
- [ ] CSV import/export works

## Common Issues

### Port Already in Use

```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

### Permission Denied

```bash
# Fix ownership
sudo chown -R perfassess:perfassess /var/www/performance-assessment

# Fix permissions
chmod 755 /var/www/performance-assessment
chmod 600 /var/www/performance-assessment/.env
```

### MongoDB Connection Failed

```bash
# Check MongoDB status
sudo systemctl status mongod

# Check authentication
mongosh -u perfassess_app -p --authenticationDatabase hr_performance

# Check logs
sudo tail -f /var/log/mongodb/mongod.log
```

### Nginx 502 Bad Gateway

```bash
# Check if Node.js is running
pm2 status

# Check Nginx error logs
sudo tail -f /var/log/nginx/performance-assessment-error.log

# Verify backend is on port 5000
sudo netstat -tlnp | grep 5000
```

## Support

For issues:
1. Check application logs: `pm2 logs`
2. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
3. Check MongoDB logs: `sudo tail -f /var/log/mongodb/mongod.log`
4. Verify firewall: `sudo ufw status`
5. Test MongoDB connection: `mongosh <connection-string>`

---

**Security Note:** Never share your `.env` file, MongoDB passwords, or JWT secret. Store backups securely and encrypt sensitive data.
