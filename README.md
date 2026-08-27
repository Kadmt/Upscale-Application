Upscale App — Browser ONNXRuntime-Web Worker

This scaffold contains TypeScript types and a worker implementation sketch for running ONNX-based image upscaling in the browser using onnxruntime-web (WebGPU preferred, WASM fallback).

Files added:
- `src/types/worker-protocol.ts` — message protocol types for worker ↔ UI
- `src/worker/inferenceWorker.ts` — WebWorker sketch to load ONNX model and run inference
- `src/utils/imageTile.ts` — tile extraction and blending helpers

Integration notes:
- Provide `onnxruntime-web` to the worker context (via `importScripts('/path/to/ort.min.js')` or bundle the worker with Vite).
- The worker expects `init` messages (with `modelUrl` or `modelArrayBuffer`) then `processImage` or `processTile` messages.
- Use transferable `ArrayBuffer`s for image buffers to minimize copies.

Development / bundling notes
- This project uses Vite. The worker is imported as an ES module (`new Worker(new URL('./worker/inferenceWorker.ts', import.meta.url), { type: 'module' })`) so Vite will bundle the worker and its dependencies (including `onnxruntime-web`).
- Put ONNX model files under `public/models/` so they are served at `/models/<name>.onnx`.
- To run locally:

```bash
npm install
npm run dev
```

Then open `http://localhost:5173` (Vite default). The app will create and bundle the worker automatically.

Notes on `onnxruntime-web` and WebGPU:
- The worker bundles `onnxruntime-web` which includes both WASM and WebGPU providers. The code will attempt to create a session with `executionProviders: ['webgpu']` and fall back where necessary.
- For production, consider hosting model files on a CDN and enabling aggressive caching.

Next steps:
- Wire these files into a React + Vite app, implement the main-thread tiling controller and UI components, and add service worker caching for model assets.

Verification checklist & SW notes

- Worker protocol: `tileResult` and `imageResult` responses include optional `durationMs` (ms to process).
- The worker imports `onnxruntime-web` directly; `init` now creates a session from the fetched model buffer.
- Full-image and tile responses use transferable `ArrayBuffer`s for performance.
- Service worker is available at `/sw.js` and will cache `/models/*` responses on-demand into `models-cache-v1`.

Service worker usage notes:
- Place ONNX models under `public/models/` so they are served at `/models/<name>.onnx`.
- To clear cached models from the page, send:

```js
if (navigator.serviceWorker?.controller) {
	navigator.serviceWorker.controller.postMessage({ action: 'clearModelsCache' });
}
```

- During development, the SW may serve cached models; unregister via browser devtools or run:

```js
navigator.serviceWorker.getRegistrations().then(regs => regs.forEach(r => r.unregister()));
```

- Verification steps to run locally:
	1. `npm install` then `npm run dev`.
	2. Open `http://localhost:5173` and upload a small image.
	3. Confirm the worker registers and `init` progress appears in console/network.
	4. Check `Network` tab → `/models/<name>.onnx` is fetched and then cached (Service Worker → Cache Storage).
	5. Observe `tileResult`/`imageResult` messages include `durationMs` and tiles render progressively.

"# Upscale-Application" 
