# URGENT: Fix External Nginx MIME Type Configuration

## Problem
The external nginx proxy at `mlapi.us` (192.168.1.196) is serving `.mjs` files with `Content-Type: application/octet-stream` instead of `text/javascript`, causing frontend loading failures on desktop browsers.

## Immediate Fix Required

### Step 1: SSH to External Server
```bash
ssh user@192.168.1.196
# (or however you access the external nginx server)
```

### Step 2: Find the nginx configuration file
```bash
# Common locations:
sudo find /etc/nginx -name "*.conf" | grep -E "(mlapi|sites-available|sites-enabled)"
# Or check:
ls -la /etc/nginx/sites-available/
ls -la /etc/nginx/sites-enabled/
```

### Step 3: Edit the mlapi.us site configuration
Add this BEFORE any existing location blocks:

```nginx
# JavaScript module files - CRITICAL FIX
location ~* \.mjs$ {
    add_header Content-Type "text/javascript";
    add_header Access-Control-Allow-Origin "*";
    proxy_pass http://192.168.1.77:3001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# WebAssembly files
location ~* \.wasm$ {
    add_header Content-Type "application/wasm";
    add_header Access-Control-Allow-Origin "*";
    proxy_pass http://192.168.1.77:3001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### Step 4: Test and Apply
```bash
# Test configuration
sudo nginx -t

# If test passes, reload nginx
sudo systemctl reload nginx
```

### Step 5: Verify Fix
```bash
curl -I https://mlapi.us/cardcam/onnx/ort-wasm-simd-threaded.jsep.mjs
```

You should see: `Content-Type: text/javascript`

## Alternative Quick Fix (if above doesn't work)

If the location blocks don't work, try adding this to the main server block:

```nginx
# Add MIME type override
location / {
    if ($request_uri ~* \.mjs$) {
        add_header Content-Type "text/javascript";
    }
    proxy_pass http://192.168.1.77:3001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

## Expected Result
After the fix, the frontend should load correctly on desktop browsers without the MIME type errors.
