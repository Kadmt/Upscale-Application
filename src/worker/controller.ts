import {
  WorkerRequest,
  WorkerResponse,
  InitRequest,
  ProcessImageRequest,
  ProcessTileRequest,
} from '../types/worker-protocol';

type ProgressHandler = (p: number, msg?: string) => void;

export class InferenceController {
  private worker: Worker;
  private onProgress?: ProgressHandler;

  constructor(workerOrUrl: string | Worker, onProgress?: ProgressHandler) {
    if (typeof workerOrUrl === 'string') {
      this.worker = new Worker(workerOrUrl, { type: 'module' });
    } else {
      this.worker = workerOrUrl;
    }
    this.onProgress = onProgress;
    this.worker.onmessage = this.handleMessage.bind(this);
  }

  private handleMessage(ev: MessageEvent) {
    const msg = ev.data as WorkerResponse;
    switch (msg.kind) {
      case 'progress':
        this.onProgress?.(msg.progress, msg.message);
        break;
      case 'sessionInfo':
        window.dispatchEvent(new CustomEvent('inference:sessionInfo', { detail: msg.info }));
        break;
      case 'tileResult':
        // Application should provide a listener to receive tiles; here we re-dispatch an event
        window.dispatchEvent(new CustomEvent('inference:tile', { detail: msg }));
        break;
      case 'imageResult':
        window.dispatchEvent(new CustomEvent('inference:image', { detail: msg }));
        break;
      case 'debug':
        // forward debug messages to the app
        window.dispatchEvent(new CustomEvent('inference:debug', { detail: msg }));
        break;
      case 'error':
        try {
          console.error('inference error', msg);
          if ((msg as any).details) console.error('inference error details', (msg as any).details);
        } catch (e) {
          // ignore
        }
        window.dispatchEvent(new CustomEvent('inference:error', { detail: msg }));
        break;
      default:
        break;
    }
  }

  async init(modelUrl: string, backend: 'webgpu' | 'wasm' = 'wasm') {
    const req: InitRequest = { kind: 'init', modelUrl, backend };
    this.worker.postMessage(req);
  }

  async processFullImage(
    taskId: string,
    imageData: ImageData,
    scale = 2,
    sharpness = 0.3,
    darkness = 0.18,
    roundness = 0.6,
    mode: 'general' | 'document8k' = 'document8k',
    tileSize?: number
  ) {
    // Simple single-message approach: transfer full image buffer to worker
    const buf = imageData.data.buffer.slice(0);
    const req: ProcessImageRequest = {
      kind: 'processImage',
      taskId,
      imageBuffer: buf,
      width: imageData.width,
      height: imageData.height,
      scale,
      sharpness,
      darkness,
      roundness,
      mode,
      tileSize,
    };
    this.worker.postMessage(req);
  }

  async processFullImageDebugY(taskId: string, imageData: ImageData, scale = 2) {
    const buf = imageData.data.buffer.slice(0);
    const req = {
      kind: 'processImage',
      taskId,
      imageBuffer: buf,
      width: imageData.width,
      height: imageData.height,
      scale,
      debugMode: 'Y',
    } as any;
    this.worker.postMessage(req, [req.imageBuffer]);
  }

  async processTiledImage(taskId: string, imageData: ImageData, scale = 2, tileSize = 512, overlap = 16) {
    // Break image into tiles and post each tile as transferable buffer
    const w = imageData.width;
    const h = imageData.height;
    const src = imageData.data;
    for (let y = 0; y < h; y += tileSize - overlap) {
      for (let x = 0; x < w; x += tileSize - overlap) {
        const tw = Math.min(tileSize, w - x);
        const th = Math.min(tileSize, h - y);
        // extract tile into a new Uint8ClampedArray
        const tile = new Uint8ClampedArray(tw * th * 4);
        let p = 0;
        for (let j = 0; j < th; j++) {
          const sy = y + j;
          for (let i = 0; i < tw; i++) {
            const sx = x + i;
            const sIdx = (sy * w + sx) * 4;
            tile[p++] = src[sIdx];
            tile[p++] = src[sIdx + 1];
            tile[p++] = src[sIdx + 2];
            tile[p++] = src[sIdx + 3];
          }
        }
        const req: ProcessTileRequest = {
          kind: 'processTile',
          taskId,
          tile: { x, y, width: tw, height: th },
          imageBuffer: tile.buffer,
          imageWidth: tw,
          imageHeight: th,
          scale,
        };
        this.worker.postMessage(req, [req.imageBuffer]);
      }
    }
  }

  // Debug helper: ask worker to render a synthetic test image
  renderTest(taskId: string, width = 256, height = 256) {
    this.worker.postMessage({ kind: 'renderTest', taskId, width, height } as any);
  }

  // Debug helper: bypass model and return a resized image from main-thread imageData
  bypassUpscale(taskId: string, imageData: ImageData, scale = 2) {
    const buf = imageData.data.buffer.slice(0);
    this.worker.postMessage({ kind: 'bypassUpscale', taskId, imageBuffer: buf, width: imageData.width, height: imageData.height, scale }, [buf]);
  }

  cancel(taskId: string) {
    this.worker.postMessage({ kind: 'cancel', taskId } as WorkerRequest);
  }

  dispose() {
    this.worker.terminate();
  }
}

// Example usage (main thread):
// const ctrl = new InferenceController('/src/worker/inferenceWorker.ts', (p,m) => console.log(p,m));
// ctrl.init('/models/real-esrgan.onnx');
// ctrl.processFullImage('task-1', imageData, 2);
