# Troubleshooting Guide - "Route Not Found" Error

## Issue: Login/Register pages work but database connection fails with "route not found"

This typically means the **frontend is loading correctly** but the **API requests are not reaching the backend**.

## Quick Diagnosis Steps

### 1. Check if Backend is Running

```bash
# Check PM2 status
pm2 status

# Should show "performance-assessment" as "online"
# If stopped, start it:
pm2 start ecosystem.config.js
```

### 2. Check Backend Logs

```bash
# View PM2 logs
pm2 logs performance-assessment --lines 50

# Look for:
# - "Server running on port 5000"
# - "MongoDB connected: hr_performance"
# - Any error messages
```

### 3. Test Backend Directly

```bash
# Test health endpoint
curl http://localhost:5000/health

# Should return: {"status":"ok","timestamp":"...","uptime":...}

# Test auth config endpoint
curl http://localhost:5000/api/auth/config

# Should return: {"allowRegistration":true}
```

If these commands fail, **the backend is not running or not accessible**.

### 4. Check Browser Console

Open browser Developer Tools (F12) → Console tab:

```
Look for errors like:
- "Failed to fetch"
- "404 Not Found"
- "ERR_CONNECTION_REFUSED"
- Network errors
```

Check the Network tab to see what URL is being called:
- Expected: `https://performance.pcconsulting.eu/api/auth/login`
- If different, the frontend API URL is wrong

## Common Causes & Fixes

### Problem 1: Backend Not Running

**Symptoms:**
- `pm2 status` shows "stopped" or "errored"
- `curl http://localhost:5000/health` fails

**Fix:**
```bash
# Check logs for errors
pm2 logs performance-assessment

# Common issues:
# - MongoDB connection failed
# - .env file not readable
# - Port 5000 already in use

# Restart backend
pm2 restart performance-assessment

# If it keeps crashing, check:
pm2 logs --err
```

### Problem 2: MongoDB Not Connected

**Symptoms:**
- Backend logs show "MongoDB connection failed"
- Backend keeps restarting

**Fix:**
```bash
# Check MongoDB status
sudo systemctl status mongod

# If not running:
sudo systemctl start mongod

# Test MongoDB connection
mongosh "mongodb://perfassess_app:YOUR_PASSWORD@localhost:27017/hr_performance?authSource=hr_performance"

# If connection fails:
# - Check MongoDB is running
# - Verify credentials in .env match MongoDB user
# - Check MongoDB authentication is enabled
```

### Problem 3: Nginx Not Proxying API Requests

**Symptoms:**
- Frontend loads fine
- Browser shows 404 for `/api/*` requests
- Direct curl to localhost:5000 works

**Fix:**

Check Nginx configuration:

```bash
# View Nginx config
sudo cat /etc/nginx/sites-enabled/performance-assessment

# Look for the /api location block
```

**Required Nginx configuration:**

```nginx
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
}
```

**If missing or incorrect:**

```bash
# Edit Nginx config
sudo vim /etc/nginx/sites-available/performance-assessment

# Add the location /api block above

# Test configuration
sudo nginx -t

# If test passes, reload Nginx
sudo systemctl reload nginx
```

### Problem 4: Wrong Frontend API URL

**Symptoms:**
- Browser console shows requests to wrong domain
- Requests go to `http://localhost:5000` instead of `https://performance.pcconsulting.eu`

**Fix:**

Check `frontend/api/apiClient.js`:

```bash
# View the file
cat /var/www/performance-assessment/frontend/api/apiClient.js | grep baseURL
```

Should show:
```javascript
this.baseURL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : 'https://performance.pcconsulting.eu/api';
```

**If it shows a different domain or localhost for production:**

```bash
# Edit the file
sudo vim /var/www/performance-assessment/frontend/api/apiClient.js

# Find the line (around line 7-9):
this.baseURL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : 'https://yourdomain.com/api';  // Change this

# Update to:
this.baseURL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : 'https://performance.pcconsulting.eu/api';

# OR use a dynamic approach:
this.baseURL = window.location.origin + '/api';
```

**After editing, clear browser cache and reload.**

### Problem 5: CORS Issues

**Symptoms:**
- Browser console shows "CORS policy" errors
- Requests are blocked

**Fix:**

Check `.env` file:

```bash
# View .env
sudo cat /var/www/performance-assessment/.env | grep FRONTEND_URL
```

Should show:
```
FRONTEND_URL=https://performance.pcconsulting.eu
```

**If incorrect:**

```bash
# Edit .env
sudo vim /var/www/performance-assessment/.env

# Update FRONTEND_URL to match your domain
FRONTEND_URL=https://performance.pcconsulting.eu

# Restart backend
pm2 restart performance-assessment
```

### Problem 6: Firewall Blocking Port 5000

**Symptoms:**
- Backend is running
- curl from server works
- Browser requests fail

**Fix:**

**This should NOT happen** because Nginx should proxy all requests.

But verify:

```bash
# Check firewall
sudo ufw status

# Port 5000 should NOT be exposed externally
# Only 22, 80, 443 should be open

# If 5000 is exposed, remove it:
sudo ufw delete allow 5000
```

**Note:** Port 5000 should only be accessible from localhost (127.0.0.1), not from the internet. Nginx proxies external requests to localhost:5000.

## Complete Diagnostic Procedure

Run these commands in order and note the results:

```bash
# 1. Check backend is running
echo "=== PM2 Status ==="
pm2 status

# 2. Check backend logs
echo "=== Backend Logs (last 20 lines) ==="
pm2 logs performance-assessment --lines 20 --nostream

# 3. Test backend locally
echo "=== Testing Backend Health ==="
curl -s http://localhost:5000/health | jq

echo "=== Testing Auth Config ==="
curl -s http://localhost:5000/api/auth/config | jq

# 4. Check MongoDB
echo "=== MongoDB Status ==="
sudo systemctl status mongod | grep Active

# 5. Check Nginx config
echo "=== Nginx API Proxy Config ==="
sudo grep -A 10 'location /api' /etc/nginx/sites-enabled/performance-assessment

# 6. Check Nginx logs
echo "=== Nginx Error Log (last 20 lines) ==="
sudo tail -20 /var/log/nginx/performance-assessment-error.log

# 7. Test Nginx proxy
echo "=== Testing via Nginx ==="
curl -s https://performance.pcconsulting.eu/api/auth/config | jq

# 8. Check .env
echo "=== .env Configuration ==="
sudo grep -E 'FRONTEND_URL|MONGODB_URI|NODE_ENV' /var/www/performance-assessment/.env

# 9. Check firewall
echo "=== Firewall Status ==="
sudo ufw status
```

## Expected Results

When everything is working:

```bash
# PM2 Status
┌─────┬──────────────────────────┬─────────┬─────────┐
│ id  │ name                     │ status  │ restart │
├─────┼──────────────────────────┼─────────┼─────────┤
│ 0   │ performance-assessment   │ online  │ 0       │
└─────┴──────────────────────────┴─────────┴─────────┘

# Backend Health
{"status":"ok","timestamp":"2026-01-08T...","uptime":123.45}

# Auth Config
{"allowRegistration":false}

# MongoDB Status
Active: active (running)

# Nginx Proxy Config shows:
location /api {
    proxy_pass http://localhost:5000;
    ...
}

# Test via Nginx
{"allowRegistration":false}
```

## Step-by-Step Fix Procedure

### If backend is not running:

```bash
# Check why it's not running
pm2 logs performance-assessment

# Common fixes:
# 1. MongoDB not running
sudo systemctl start mongod

# 2. .env file permissions
sudo chown perfassess:perfassess /var/www/performance-assessment/.env
sudo chmod 600 /var/www/performance-assessment/.env

# 3. Wrong MongoDB credentials
sudo vim /var/www/performance-assessment/.env
# Update MONGODB_URI

# 4. Start backend
pm2 start ecosystem.config.js
pm2 save
```

### If Nginx is not proxying:

```bash
# 1. Create/update Nginx config with /api location block
sudo vim /etc/nginx/sites-available/performance-assessment

# 2. Test configuration
sudo nginx -t

# 3. Reload Nginx
sudo systemctl reload nginx

# 4. Check Nginx error logs
sudo tail -f /var/log/nginx/performance-assessment-error.log
```

### If frontend API URL is wrong:

```bash
# 1. Update apiClient.js
sudo vim /var/www/performance-assessment/frontend/api/apiClient.js

# Change to use window.location.origin:
this.baseURL = window.location.origin + '/api';

# 2. Clear browser cache
# 3. Hard reload (Ctrl+Shift+R)
```

## Testing After Fixes

1. **Test backend directly:**
```bash
curl http://localhost:5000/api/auth/config
```

2. **Test through Nginx:**
```bash
curl https://performance.pcconsulting.eu/api/auth/config
```

3. **Test in browser:**
- Open https://performance.pcconsulting.eu/login.html
- Open Developer Tools (F12) → Network tab
- Try to login
- Check if request to `/api/auth/login` succeeds

## Still Not Working?

Collect this information:

```bash
# Save diagnostic info to file
cat > /tmp/diagnostic.txt << 'EOF'
=== PM2 Status ===
EOF
pm2 status >> /tmp/diagnostic.txt

echo "" >> /tmp/diagnostic.txt
echo "=== Backend Logs ===" >> /tmp/diagnostic.txt
pm2 logs performance-assessment --lines 50 --nostream >> /tmp/diagnostic.txt

echo "" >> /tmp/diagnostic.txt
echo "=== Backend Health ===" >> /tmp/diagnostic.txt
curl -s http://localhost:5000/health >> /tmp/diagnostic.txt

echo "" >> /tmp/diagnostic.txt
echo "=== Nginx Config ===" >> /tmp/diagnostic.txt
sudo cat /etc/nginx/sites-enabled/performance-assessment >> /tmp/diagnostic.txt

echo "" >> /tmp/diagnostic.txt
echo "=== .env (sanitized) ===" >> /tmp/diagnostic.txt
sudo grep -E 'NODE_ENV|PORT|FRONTEND_URL' /var/www/performance-assessment/.env >> /tmp/diagnostic.txt

# View the diagnostic file
cat /tmp/diagnostic.txt
```

Share this output for further troubleshooting.

## Quick Reference Commands

```bash
# Restart everything
sudo systemctl restart mongod
pm2 restart performance-assessment
sudo systemctl reload nginx

# View logs
pm2 logs performance-assessment
sudo tail -f /var/log/nginx/performance-assessment-error.log
sudo tail -f /var/log/mongodb/mongod.log

# Test endpoints
curl http://localhost:5000/health
curl http://localhost:5000/api/auth/config
curl https://performance.pcconsulting.eu/api/auth/config

# Check processes
pm2 status
sudo systemctl status mongod
sudo systemctl status nginx

# Check ports
sudo netstat -tlnp | grep 5000
sudo netstat -tlnp | grep 27017
```

## Most Likely Causes (in order)

1. **Nginx not proxying `/api` requests** → Add location block
2. **Backend not running** → Check PM2, start backend
3. **MongoDB not connected** → Check MongoDB status, credentials
4. **Wrong frontend API URL** → Update apiClient.js
5. **.env file issues** → Check permissions, values

Start with checking Nginx configuration - this is the most common cause of "route not found" when the frontend loads but API calls fail.
