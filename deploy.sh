#!/bin/bash

################################################################################
# Production Deployment Script for Performance Assessment Application
# For Ubuntu Server 20.04/22.04
#
# Usage: sudo ./deploy.sh
#
# This script will:
# - Install all required dependencies
# - Set up MongoDB
# - Configure Nginx
# - Set up PM2 for process management
# - Configure firewall
# - Set up SSL with Let's Encrypt
#
# WARNING: Review this script before running!
################################################################################

set -e  # Exit on error

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/var/www/performance-assessment"
APP_USER="perfassess"
NODE_VERSION="18"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Performance Assessment Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Ask for confirmation
read -p "This will install and configure the entire application. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Deployment cancelled."
    exit 1
fi

echo -e "${YELLOW}Step 1: Updating system...${NC}"
apt update && apt upgrade -y

echo -e "${YELLOW}Step 2: Installing Node.js ${NODE_VERSION}...${NC}"
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
    apt install -y nodejs
    echo -e "${GREEN}✓ Node.js installed: $(node --version)${NC}"
else
    echo -e "${GREEN}✓ Node.js already installed: $(node --version)${NC}"
fi

echo -e "${YELLOW}Step 3: Installing MongoDB...${NC}"
if ! command -v mongod &> /dev/null; then
    curl -fsSL https://pgp.mongodb.com/server-8.0.asc | \
       gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor

    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | \
       tee /etc/apt/sources.list.d/mongodb-org-8.0.list

    apt update
    apt install -y mongodb-org

    systemctl start mongod
    systemctl enable mongod

    echo -e "${GREEN}✓ MongoDB installed and started${NC}"
else
    echo -e "${GREEN}✓ MongoDB already installed${NC}"
fi

echo -e "${YELLOW}Step 4: Installing Nginx...${NC}"
if ! command -v nginx &> /dev/null; then
    apt install -y nginx
    systemctl start nginx
    systemctl enable nginx
    echo -e "${GREEN}✓ Nginx installed and started${NC}"
else
    echo -e "${GREEN}✓ Nginx already installed${NC}"
fi

echo -e "${YELLOW}Step 5: Installing PM2...${NC}"
if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
    echo -e "${GREEN}✓ PM2 installed${NC}"
else
    echo -e "${GREEN}✓ PM2 already installed${NC}"
fi

echo -e "${YELLOW}Step 6: Installing Certbot...${NC}"
if ! command -v certbot &> /dev/null; then
    apt install -y certbot python3-certbot-nginx
    echo -e "${GREEN}✓ Certbot installed${NC}"
else
    echo -e "${GREEN}✓ Certbot already installed${NC}"
fi

echo -e "${YELLOW}Step 7: Creating application user...${NC}"
if ! id "$APP_USER" &>/dev/null; then
    useradd -m -s /bin/bash $APP_USER
    echo -e "${GREEN}✓ User $APP_USER created${NC}"
else
    echo -e "${GREEN}✓ User $APP_USER already exists${NC}"
fi

echo -e "${YELLOW}Step 8: Setting up application directory...${NC}"
mkdir -p $APP_DIR
chown $APP_USER:$APP_USER $APP_DIR

if [ -d ".git" ]; then
    echo "Copying application files..."
    cp -r . $APP_DIR/
    chown -R $APP_USER:$APP_USER $APP_DIR
    echo -e "${GREEN}✓ Application files copied${NC}"
else
    echo -e "${YELLOW}⚠ Not a git repository. Please manually copy files to $APP_DIR${NC}"
fi

echo -e "${YELLOW}Step 9: Installing application dependencies...${NC}"
cd $APP_DIR
sudo -u $APP_USER npm install --production
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${YELLOW}Step 10: Creating .env file template...${NC}"
if [ ! -f "$APP_DIR/.env" ]; then
    cat > $APP_DIR/.env << EOF
# Server Configuration
NODE_ENV=production
PORT=5000
# FRONTEND_URL=https://yourdomain.com
FRONTEND_URL=https://performance.pcconsulting.eu

# MongoDB Configuration
MONGODB_URI=mongodb://perfassess_app:CHANGE_THIS_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance
MONGODB_DB_NAME=hr_performance

# JWT Configuration
JWT_SECRET=$(node -e "console.log(require('crypto').randomBytes(64).toString('hex'))")
JWT_EXPIRES_IN=24h

# Security
BCRYPT_ROUNDS=12

# Registration Control
ALLOW_REGISTRATION=false
EOF

    chmod 600 $APP_DIR/.env
    chown $APP_USER:$APP_USER $APP_DIR/.env

    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ IMPORTANT: Edit $APP_DIR/.env with your actual values!${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

echo -e "${YELLOW}Step 11: Creating PM2 ecosystem file...${NC}"
if [ ! -f "$APP_DIR/ecosystem.config.js" ]; then
    cat > $APP_DIR/ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'performance-assessment',
    script: './backend/server.js',
    instances: 2,
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
    watch: false
  }]
};
EOF
    chown $APP_USER:$APP_USER $APP_DIR/ecosystem.config.js
    echo -e "${GREEN}✓ PM2 ecosystem file created${NC}"
else
    echo -e "${GREEN}✓ PM2 ecosystem file already exists${NC}"
fi

echo -e "${YELLOW}Step 12: Creating log directory...${NC}"
mkdir -p /var/log/performance-assessment
chown $APP_USER:$APP_USER /var/log/performance-assessment
echo -e "${GREEN}✓ Log directory created${NC}"

echo -e "${YELLOW}Step 13: Configuring firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw --force enable
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp comment 'SSH'
    ufw allow 80/tcp comment 'HTTP'
    ufw allow 3000/tcp comment 'ALT_HTTP'
    ufw allow 443/tcp comment 'HTTPS'
    echo -e "${GREEN}✓ Firewall configured${NC}"
else
    echo -e "${YELLOW}⚠ UFW not found. Install with: apt install ufw${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Configure MongoDB:"
echo "   sudo mongosh"
echo "   use hr_performance"
echo "   db.createUser({user: 'perfassess_app', pwd: 'YOUR_PASSWORD', roles: [{role: 'readWrite', db: 'hr_performance'}]})"
echo ""
echo "2. Update .env file with your configuration:"
echo "   sudo vim $APP_DIR/.env"
echo "   - Update FRONTEND_URL with your domain"
echo "   - Update MONGODB_URI with MongoDB password"
echo ""
echo "3. Start the application:"
echo "   sudo -u $APP_USER pm2 start $APP_DIR/ecosystem.config.js"
echo "   sudo -u $APP_USER pm2 save"
echo "   sudo env PATH=\$PATH:/usr/bin pm2 startup systemd -u $APP_USER --hp /home/$APP_USER"
echo ""
echo "4. Configure Nginx (see DEPLOYMENT.md for full config)"
echo "   sudo vim /etc/nginx/sites-available/performance-assessment"
echo "   sudo ln -s /etc/nginx/sites-available/performance-assessment /etc/nginx/sites-enabled/"
echo "   sudo nginx -t"
echo "   sudo systemctl reload nginx"
echo ""
echo "5. Get SSL certificate:"
# echo "   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com"
echo "   sudo certbot --nginx -d performance.pcconsulting.eu -d www.performance.pcconsulting.eu"
echo ""
echo "6. Test your application:"
# echo "   https://yourdomain.com"
echo "   https://performance.pcconsulting.eu"
echo ""
echo -e "${GREEN}For detailed instructions, see DEPLOYMENT.md${NC}"
echo ""
