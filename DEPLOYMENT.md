# CardCam Frontend Deployment Guide

## Architecture

```
Browser → Production Nginx (mlapi.us) → Frontend Container (npm start)
          proxy_pass to                  192.168.1.77:3001
          192.168.1.77:3001
          adds CORS headers
```

Simple! The production nginx just proxies to the React dev server running in Docker.

## Deployment Steps

### Step 1: Start Frontend Container

```bash
cd /home/alec/git/pokemon
docker compose up -d --build frontend
```

This starts the React dev server on port 3001 (mapped from container port 3000).

### Step 2: Update Production Nginx Config

Copy `production_server_nginx.conf` to your production server:

```bash
scp production_server_nginx.conf alec@mlapi.us:~/

# SSH to server
ssh alec@mlapi.us

# Backup current config
sudo cp /etc/nginx/sites-available/mlapi.us /etc/nginx/sites-available/mlapi.us.backup

# Replace with new config
sudo mv ~/production_server_nginx.conf /etc/nginx/sites-available/mlapi.us

# Test config
sudo nginx -t

# If test passes, reload
sudo systemctl reload nginx
```

### Step 3: Verify

1. Open https://mlapi.us/cardcam/ in browser
2. Open console (F12)
3. Look for:

```
Cross-origin isolation: true  ← MUST BE TRUE!
✅ WebGPU: CONFIRMED - Actually using GPU acceleration
```

## Key Configuration

### Frontend Container (docker-compose.yml)
- Runs `npm start` (React dev server)
- Exposes port 3001
- Hot reload enabled with volume mounts

### Production Nginx (production_server_nginx.conf)
- Proxies `/cardcam/` to `http://192.168.1.77:3001/`
- Adds cross-origin isolation headers
- Supports WebSocket for hot reload

## Troubleshooting

### Cross-origin isolation: false

Check nginx headers:
```bash
curl -I https://mlapi.us/cardcam/ | grep -i cross-origin
```

Should see:
```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
```

### Container not running

```bash
docker ps | grep card-scanner-frontend
docker logs card-scanner-frontend
```

### Port 3001 not accessible

```bash
# On the machine running docker
curl http://localhost:3001
```

### WebGPU not working

1. Verify cross-origin isolation is enabled (`true`)
2. Check browser console for errors
3. Verify you're using `/webgpu` ONNX Runtime build
4. Check CSP allows `blob:` workers

## Local Development

If you want to develop locally without Docker:

```bash
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

Then update your `/etc/hosts` or just access via localhost.

## Files

- **`frontend/Dockerfile`** - Runs `npm start` in container
- **`docker-compose.yml`** - Frontend service config
- **`production_server_nginx.conf`** - Complete production nginx config
- **`frontend/nginx.conf`** - Not used (can be deleted)

## Summary

**No nginx in the container!** 
- Container runs React dev server
- Production nginx proxies to it
- Single source of truth for headers
- Simple and clean!
