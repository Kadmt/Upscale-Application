// Worker message protocol types for the inference worker
export type Backend = 'webgpu' | 'wasm';

export interface InitRequest {
  kind: 'init';
  modelUrl: string; // URL to fetch the ONNX model (can be relative to public/)
  backend?: Backend; // preferred backend
  fp16?: boolean; // hint to prefer FP16/quantized model if available
}

export interface WarmupRequest {
  kind: 'warmup';
  tileSize?: number;
}

// Image tile is sent as a transferable ArrayBuffer (RGBA uint8) alongside metadata
export interface ProcessTileRequest {
  kind: 'processTile';
  taskId: string; // opaque id for this task
  tile: { x: number; y: number; width: number; height: number };
  imageBuffer: ArrayBuffer; // Uint8Clamped RGBA data for the tile
  imageWidth: number; // width of the tile image buffer
  imageHeight: number;
  scale: number; // upscale factor (e.g., 2)
  settings?: Record<string, any>;
}

export interface ProcessImageRequest {
  kind: 'processImage';
  taskId: string;
  imageBuffer: ArrayBuffer; // full-image RGBA uint8 buffer
  width: number;
  height: number;
  scale: number;
  tileSize?: number; // optional hint
  sharpness?: number; // 0..2.5 enhancement strength
  darkness?: number; // 0..0.5 smooth text darkening factor
  mode?: 'general' | 'document8k'; // enhancement profile
}

export interface CancelRequest {
  kind: 'cancel';
  taskId: string;
}

export type WorkerRequest =
  | InitRequest
  | WarmupRequest
  | ProcessTileRequest
  | ProcessImageRequest
  | CancelRequest;

// Responses posted back from worker
export interface ProgressResponse {
  kind: 'progress';
  taskId?: string;
  progress: number; // 0..1
  message?: string;
}

export interface TileResultResponse {
  kind: 'tileResult';
  taskId: string;
  tile: { x: number; y: number; width: number; height: number };
  imageBuffer: ArrayBuffer; // RGBA uint8 for the upscaled tile
  width: number; // width of returned buffer
  height: number;
  durationMs?: number; // optional processing time in milliseconds
}

export interface ImageResultResponse {
  kind: 'imageResult';
  taskId: string;
  imageBuffer: ArrayBuffer; // RGBA uint8 for the full upscaled image
  width: number;
  height: number;
  durationMs?: number; // optional processing time in milliseconds
}

export interface ErrorResponse {
  kind: 'error';
  taskId?: string;
  message: string;
}

export type WorkerResponse =
  | ProgressResponse
  | TileResultResponse
  | ImageResultResponse
  | ErrorResponse;
