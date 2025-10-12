# CardCam Frontend

React-based frontend for the trading card scanner with WebGPU/WASM acceleration.

## Local Development

```bash
# Install dependencies
npm install

# Run development server
npm start
# Opens at http://localhost:3000

# Build for production
npm run build
# Creates build/ directory with static files
```

## Production Deployment

**Do NOT use Docker for production.** The production server serves static files directly.

### Deploy to Production Server

```bash
# From project root:
./deploy-cardcam.sh
```

This will:
1. Build the React app (`npm run build`)
2. Create a tarball
3. Upload to production server (mlapi.us)
4. Extract to `/var/www/cardcam/`
5. Set correct permissions

Then update your nginx config on the server to serve files from `/var/www/cardcam/`.

See `../DEPLOYMENT.md` for complete instructions.

## Why No Docker Container?

We removed the nginx container because:
- Production server serves static files directly (faster, simpler)
- No need for a running container just to serve static files
- Eliminates header conflicts between two nginx instances
- Cleaner separation: build locally, deploy files to server

## Architecture

```
Development:  npm start → http://localhost:3000 (dev server)
Production:   nginx → /var/www/cardcam/ (static files)
```

## Key Features

- WebGPU acceleration for ML inference (6x faster than WASM)
- Real-time trading card detection
- Camera interface with live preview
- Card extraction and identification
- Cross-origin isolation for WebAssembly threading

## Configuration

- **Base URL:** `/cardcam/` (configured in `package.json` homepage field)
- **Model Path:** `/cardcam/models/trading_card_detector.onnx`
- **ONNX Runtime:** `/cardcam/onnx/` (WASM files)

## Troubleshooting

### Cross-origin isolation not enabled

Make sure production nginx has these headers:
```nginx
add_header Cross-Origin-Embedder-Policy "require-corp" always;
add_header Cross-Origin-Opener-Policy "same-origin" always;
```

### WebGPU not working

Check browser console for:
- `Cross-origin isolation: true` (must be true)
- `ort.env.webgpu available: true`
- `✅ WebGPU: CONFIRMED`

If you see fallback errors, check the nginx config matches `production_server_nginx.conf`.



