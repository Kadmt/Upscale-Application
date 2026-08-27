/* eslint-disable no-restricted-globals */
// WebWorker for running ONNXRuntime-Web inference (WebGPU preferred, WASM fallback)
import * as ort from 'onnxruntime-web'

// Configure WASM static asset paths & log level for ONNX Runtime Web
if (ort?.env) {
  ort.env.logLevel = 'error';
  if (ort.env.wasm) {
    const origin = typeof self !== 'undefined' && self.location && self.location.origin ? self.location.origin : '';
    ort.env.wasm.wasmPaths = origin + '/ort-files/';
  }
}

type AnyObject = Record<string, any>;

let session: any = null;
let currentTasks = new Set<string>();

function resolveMetadataByName(metaObj: any, name?: string) {
  if (!metaObj) return undefined;
  const normalize = (m: any) => {
    if (!m) return m;
    if (!m.dims && m.shape) m.dims = m.shape;
    return m;
  };
  // If metadata is an array of entries (some runtimes return arrays)
  if (Array.isArray(metaObj)) {
    if (name) {
      for (const item of metaObj) {
        if (!item) continue;
        if (item.name === name || item.name === String(name)) return normalize(item);
      }
    }
    return normalize(metaObj[0]);
  }
  // object mapping
  if (name && metaObj[name]) return normalize(metaObj[name]);
  // try numeric index property
  if (metaObj[0]) return normalize(metaObj[0]);
  const keys = Object.keys(metaObj);
  if (keys.length) return normalize(metaObj[keys[0]]);
  return undefined;
}

function post<T>(msg: T, transfer?: Transferable[]) {
  ;(self as any).postMessage(msg, transfer || []);
}

async function initSession(modelArrayBuffer: ArrayBuffer, backend: 'webgpu' | 'wasm' = 'wasm') {
  // Configure WASM static asset paths via absolute origin URL
  if (ort?.env?.wasm) {
    const origin = typeof self !== 'undefined' && self.location && self.location.origin ? self.location.origin : '';
    ort.env.wasm.wasmPaths = origin + '/ort-files/';
  }

  // Try execution providers in order: backend -> wasm (multi-thread) -> wasm (single-thread)
  try {
    const opts: AnyObject = { executionProviders: [backend] };
    session = await ort.InferenceSession.create(modelArrayBuffer, opts);
  } catch (err1) {
    if (backend !== 'wasm') {
      try {
        session = await ort.InferenceSession.create(modelArrayBuffer, { executionProviders: ['wasm'] });
      } catch (err2) {
        if (ort?.env?.wasm) {
          ort.env.wasm.numThreads = 1;
        }
        session = await ort.InferenceSession.create(modelArrayBuffer, { executionProviders: ['wasm'] });
      }
    } else {
      if (ort?.env?.wasm) {
        ort.env.wasm.numThreads = 1;
      }
      session = await ort.InferenceSession.create(modelArrayBuffer, { executionProviders: ['wasm'] });
    }
  }

  if (session) {
    try {
      session.inputNames = session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : undefined);
      session.outputNames = session.outputNames || (session.outputMetadata ? Object.keys(session.outputMetadata) : undefined);
    } catch (e) {}
  }
  return session;
}

function uint8ToFloat32RGB(src: Uint8ClampedArray) {
  const len = (src.length / 4) * 3; // RGB per pixel
  const out = new Float32Array(len);
  let j = 0;
  for (let i = 0; i < src.length; i += 4) {
    out[j++] = src[i] / 255; // R
    out[j++] = src[i + 1] / 255; // G
    out[j++] = src[i + 2] / 255; // B
  }
  return out;
}

function uint8ToFloat32Y(src: Uint8ClampedArray) {
  const len = src.length / 4; // one Y per pixel
  const out = new Float32Array(len);
  let j = 0;
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i] / 255;
    const g = src[i + 1] / 255;
    const b = src[i + 2] / 255;
    // Y channel conversion (ITU-R BT.709 sRGB)
    out[j++] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return out;
}

// Convert NHWC float array to tensor depending on model expected layout
function makeInputTensor(data: Float32Array, width: number, height: number, channels = 3) {
  // create NCHW float tensor: [1,C,H,W]
  const C = channels;
  const H = height;
  const W = width;
  const tensorData = new Float32Array(1 * C * H * W);
  if (C === 3) {
    let idx = 0;
    for (let c = 0; c < C; c++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          tensorData[idx++] = data[(h * W + w) * 3 + c];
        }
      }
    }
  } else if (C === 1) {
    // data is single-channel per pixel
    let idx = 0;
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        tensorData[idx++] = data[h * W + w];
      }
    }
  }
  return new ort.Tensor('float32', tensorData, [1, C, H, W]);
}

function makeNHWCTensor(data: Float32Array, width: number, height: number, channels = 3) {
  // data is expected as [R,G,B, R,G,B, ...] per pixel (NHWC flatten)
  const H = height;
  const W = width;
  const C = channels;
  return new ort.Tensor('float32', data, [1, H, W, C]);
}

function buildShapeFromMeta(meta: any, channels: number, H: number, W: number) {
  if (!meta || !meta.dims) {
    return [1, channels, H, W];
  }
  const dims = meta.dims.slice();
  // If dims length 4, assume [N,C,H,W] and replace -1/0 with actuals
  if (dims.length === 4) {
    const out = dims.map((d: number, i: number) => {
      if (d > 0) return d;
      // replace dynamic dims: N->1, C->channels, H->H, W->W
      if (i === 0) return 1;
      if (i === 1) return channels;
      if (i === 2) return H;
      return W;
    });
    return out;
  }
  // If dims length 3, could be [C,H,W]
  if (dims.length === 3) {
    const [c, hh, ww] = dims;
    const C = c > 0 ? c : channels;
    const H2 = hh > 0 ? hh : H;
    const W2 = ww > 0 ? ww : W;
    return [1, C, H2, W2];
  }
  // Fallback to [1,channels,H,W]
  return [1, channels, H, W];
}

async function runInferenceOnTile(tileBuffer: ArrayBuffer, tileWidth: number, tileHeight: number) {
  if (!session) throw new Error('Session not initialized');
  // Debug: report incoming tile and session expectations
  try {
    const expectedInputNames = session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : ['input']);
    const inputMeta = resolveMetadataByName(session.inputMetadata, expectedInputNames[0]);
    post({ kind: 'debug', message: 'runInferenceOnTile start', tileWidth, tileHeight, inputMeta });
  } catch (e) {}
  const u8 = new Uint8ClampedArray(tileBuffer);
  // determine expected input channels from session metadata when possible
  const expectedInputNames = session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : ['input']);
  const inputMeta = session.inputMetadata ? session.inputMetadata[expectedInputNames[0]] : undefined;
  // default to 1 channel for many SR models (Y channel); fall back to metadata when available
  const expectedChannels = inputMeta && inputMeta.dims ? (Number(inputMeta.dims[1]) || 1) : 1;
  let input: any;
  // helper: resize RGBA tile to dstW/dstH using OffscreenCanvas
  function resizeRGBA(u8array: Uint8ClampedArray, srcW: number, srcH: number, dstW: number, dstH: number) {
    if (srcW === dstW && srcH === dstH) return u8array;
    try {
      const srcCanvas = new OffscreenCanvas(srcW, srcH);
      const sctx = srcCanvas.getContext('2d');
      const img = new ImageData(new Uint8ClampedArray(u8array), srcW, srcH);
      sctx.putImageData(img, 0, 0);
      const dstCanvas = new OffscreenCanvas(dstW, dstH);
      const dctx = dstCanvas.getContext('2d');
      dctx.drawImage(srcCanvas, 0, 0, dstW, dstH);
      const out = dctx.getImageData(0, 0, dstW, dstH);
      return out.data;
    } catch (e) {
      // fallback: no resize
      return u8array;
    }
  }
  // determine target dims from metadata (if fixed) or use tile dims
  // determine target dims from metadata (if fixed). Prefer model dims; fallback to 224.
  let targetH = 224;
  let targetW = 224;
  try {
    const findMeta = inputMeta || (session.inputMetadata && session.inputMetadata[0]);
    if (findMeta && findMeta.dims) {
      const dims = findMeta.dims;
      const maybeH = Number(dims[2]);
      const maybeW = Number(dims[3]);
      if (maybeH > 0) targetH = maybeH;
      if (maybeW > 0) targetW = maybeW;
    }
  } catch (e) {
    // ignore and use defaults
  }

  if (expectedChannels === 1) {
    // possibly resize RGBA tile to expected H/W, then convert to Y
    const usedU8 = (targetW !== tileWidth || targetH !== tileHeight) ? resizeRGBA(u8, tileWidth, tileHeight, targetW, targetH) : u8;
    const floatData = uint8ToFloat32Y(usedU8 as Uint8ClampedArray);
    input = makeInputTensor(floatData, targetW, targetH, 1);
  } else {
    const usedU8 = (targetW !== tileWidth || targetH !== tileHeight) ? resizeRGBA(u8, tileWidth, tileHeight, targetW, targetH) : u8;
    const floatData = uint8ToFloat32RGB(usedU8 as Uint8ClampedArray);
    input = makeInputTensor(floatData, targetW, targetH, 3);
  }
  // determine canonical input name
  const inputName = (session.inputNames && session.inputNames[0]) || (session.inputMetadata ? Object.keys(session.inputMetadata)[0] : 'input') || 'input';
  const meta = session.inputMetadata ? (session.inputMetadata[inputName] || session.inputMetadata[0]) : undefined;
  const nchwShape = buildShapeFromMeta(meta, expectedChannels, targetH, targetW);
  const nhwcShape = [1, targetH, targetW, expectedChannels];
  // zero-probe NCHW
  try {
    const zeroNCHW = new ort.Tensor('float32', new Float32Array(nchwShape.reduce((a:number,b:number)=>a*b,1)).fill(0), nchwShape);
    post({ kind: 'debug', message: `probing input='${inputName}' nchw=${nchwShape.join('x')}` });
    const probeFeeds: AnyObject = {};
    probeFeeds[inputName] = zeroNCHW;
    await session.run(probeFeeds);
    post({ kind: 'progress', progress: 0.6, message: `probe succeeded with feed='${inputName}' layout=NCHW` });
  } catch (probeErr: any) {
    post({ kind: 'progress', message: `zero-probe failed for '${inputName}': ${probeErr?.message || String(probeErr)}` });
  }

  // try real run NCHW
  let output: any = null;
  try {
    // Use the previously prepared `input` tensor (resized and converted) to avoid data-length mismatch
    const tensorToRun = input || makeInputTensor(expectedChannels === 1 ? uint8ToFloat32Y(u8) : uint8ToFloat32RGB(u8), targetW, targetH, expectedChannels);
    post({ kind: 'debug', message: `running session.run with feed='${inputName}' shape=${tensorToRun.dims ? tensorToRun.dims.join('x') : '[unknown]'} len=${tensorToRun.data.length}` });
    const feeds: AnyObject = {};
    feeds[inputName] = tensorToRun;
    output = await session.run(feeds);
    post({ kind: 'progress', progress: 0.9, message: `session.run succeeded with feed='${inputName}' layout=NCHW` });
  } catch (errN: any) {
    // try NHWC if multichannel
    if (expectedChannels > 1) {
      try {
        const realNHWCData = uint8ToFloat32RGB(u8);
        const realNHWCTensor = makeNHWCTensor(realNHWCData, targetW, targetH, expectedChannels);
        const feeds2: AnyObject = {};
        feeds2[inputName] = realNHWCTensor;
        post({ kind: 'debug', message: `running session.run with feed='${inputName}' NHWC shape=${realNHWCTensor.dims.join('x')}` });
        output = await session.run(feeds2);
        post({ kind: 'progress', progress: 0.9, message: `session.run succeeded with feed='${inputName}' layout=NHWC` });
      } catch (errH: any) {
        const debug: any = { error: 'all attempts failed', attempts: [{ candidate: inputName, layout: 'NCHW', message: errN?.message || String(errN) }, { candidate: inputName, layout: 'NHWC', message: errH?.message || String(errH) }], sessionInputMetadata: session.inputMetadata, sessionOutputMetadata: session.outputMetadata, candidates: [inputName] };
        post({ kind: 'error', message: `failed to call OrtRun(). All attempts failed.`, details: debug });
        throw new Error('OrtRun failed for input name ' + inputName);
      }
    } else {
      const debug: any = { error: 'all attempts failed', attempts: [{ candidate: inputName, layout: 'NCHW', message: errN?.message || String(errN) }], sessionInputMetadata: session.inputMetadata, sessionOutputMetadata: session.outputMetadata, candidates: [inputName] };
      post({ kind: 'error', message: `failed to call OrtRun(). All attempts failed.`, details: debug });
      throw new Error('OrtRun failed for input name ' + inputName);
    }
  }
  const outputName = session.outputNames ? session.outputNames[0] : Object.keys(output)[0];
  const outTensor = output[outputName];
  // Convert output tensor (NCHW) back to Uint8 RGBA buffer (simple conversion)
  const [n, c, h, w] = outTensor.dims;
  const outData = outTensor.data as Float32Array;
  const outU8 = new Uint8ClampedArray(w * h * 4);
  let p = 0;
  if (c === 1) {
    // duplicate Y channel to RGB
    for (let i = 0; i < w * h; i++) {
      const y = outData[i];
      const v = Math.max(0, Math.min(255, Math.round(y * 255)));
      outU8[p++] = v;
      outU8[p++] = v;
      outU8[p++] = v;
      outU8[p++] = 255;
    }
  } else {
    for (let i = 0; i < w * h; i++) {
      const r = outData[i * 3 + 0];
      const g = outData[i * 3 + 1];
      const b = outData[i * 3 + 2];
      outU8[p++] = Math.max(0, Math.min(255, Math.round(r * 255)));
      outU8[p++] = Math.max(0, Math.min(255, Math.round(g * 255)));
      outU8[p++] = Math.max(0, Math.min(255, Math.round(b * 255)));
      outU8[p++] = 255;
    }
  }
  try {
    // debug samples
    const sampleOut = Array.from(outData.slice(0, 20));
    const sampleRGBA = Array.from(outU8.slice(0, 40));
    let minU = 255, maxU = 0;
    for (let i = 0; i < outU8.length; i++) {
      const v = outU8[i];
      if (v < minU) minU = v;
      if (v > maxU) maxU = v;
    }
    post({ kind: 'debug', message: 'runInferenceOnTile final samples', sampleOut, sampleRGBA, minU, maxU });
  } catch (e) {}
  return { buffer: outU8.buffer, width: w, height: h };
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as any;
  try {
    if (!msg || !msg.kind) return;
    switch (msg.kind) {
      case 'renderTest': {
        // create a simple gradient RGBA image for debugging display pipeline
        const w = msg.width || 256;
        const h = msg.height || 256;
        const buf = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            buf[i] = (x / w) * 255; // R gradient
            buf[i + 1] = (y / h) * 255; // G gradient
            buf[i + 2] = 128; // B constant
            buf[i + 3] = 255;
          }
        }
        post({ kind: 'imageResult', taskId: msg.taskId || 'renderTest', imageBuffer: buf.buffer, width: w, height: h }, [buf.buffer]);
        break;
      }
      case 'init': {
        post({ kind: 'progress', progress: 0.01, message: 'loading runtime' });
        // msg.modelUrl is expected to be provided as ArrayBuffer or URL
        let modelBuffer: ArrayBuffer;
        if (msg.modelArrayBuffer) {
          modelBuffer = msg.modelArrayBuffer;
        } else if (msg.modelUrl) {
          const origin = typeof self !== 'undefined' && self.location && self.location.origin ? self.location.origin : '';
          const fullUrl = new URL(msg.modelUrl, origin || location.origin).href;
          const r = await fetch(fullUrl);
          if (!r.ok) {
            throw new Error(`Failed to fetch model from ${fullUrl} (status: ${r.status})`);
          }
          modelBuffer = await r.arrayBuffer();
          if (!modelBuffer || modelBuffer.byteLength === 0) {
            throw new Error(`Model file at ${fullUrl} is empty (0 bytes).`);
          }
        } else {
          throw new Error('init requires modelUrl or modelArrayBuffer');
        }
        const backend = msg.backend || (typeof (self as any).navigator !== 'undefined' && (navigator as any).gpu ? 'webgpu' : 'wasm');
        post({ kind: 'progress', progress: 0.1, message: `initializing session (${backend})` });
        try {
          session = await initSession(modelBuffer, backend);
        } catch (initErr: any) {
          post({ kind: 'progress', progress: 0.15, message: `session init failed for backend=${backend}: ${initErr?.message || String(initErr)}` });
          // automatic fallback to wasm for debugging if webgpu failed
          if (backend === 'webgpu') {
            try {
              post({ kind: 'progress', progress: 0.16, message: 'falling back to wasm backend for init' });
              session = await initSession(modelBuffer, 'wasm');
              post({ kind: 'progress', progress: 0.2, message: 'session initialized with wasm fallback' });
            } catch (wasmErr: any) {
              post({ kind: 'error', message: `both webgpu and wasm init failed: ${wasmErr?.message || String(wasmErr)}` });
              throw wasmErr;
            }
          } else {
            throw initErr;
          }
        }
              // expose session info to main thread for debugging
              try {
                const info: any = {
                  inputNames: session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : []),
                  inputMetadata: session.inputMetadata,
                  outputNames: session.outputNames || (session.outputMetadata ? Object.keys(session.outputMetadata) : []),
                  outputMetadata: session.outputMetadata,
                };
                // build a human-friendly string summarizing dims/types for quick copy/paste
                try {
                  const readable: any = { inputs: {}, outputs: {} };
                  if (info.inputMetadata) {
                    for (const k of Object.keys(info.inputMetadata)) {
                      const m = info.inputMetadata[k];
                      readable.inputs[k] = { type: m.type, dims: m.dims };
                    }
                  }
                  if (info.outputMetadata) {
                    for (const k of Object.keys(info.outputMetadata)) {
                      const m = info.outputMetadata[k];
                      readable.outputs[k] = { type: m.type, dims: m.dims };
                    }
                  }
                  info.human = JSON.stringify(readable, null, 2);
                } catch (ex) {
                  info.human = undefined;
                }
                post({ kind: 'sessionInfo', info });
              } catch (e) {
                // ignore
              }
              post({ kind: 'progress', progress: 0.5, message: 'session ready' });
        break;
      }
      case 'warmup': {
        if (!session) throw new Error('session not ready');
        // run a small dummy tensor to warm-up WebGPU pipeline
        const size = msg.tileSize || 32;
        const dummy = new Float32Array(size * size * 3).fill(0.5);
        const t = makeInputTensor(dummy, size, size);
        const feeds: AnyObject = {};
        const inputNames = session.inputNames || ['input'];
        feeds[inputNames[0]] = t;
        await session.run(feeds);
        post({ kind: 'progress', progress: 0.7, message: 'warmup done' });
        break;
      }
      case 'processTile': {
        currentTasks.add(msg.taskId);
        post({ kind: 'progress', taskId: msg.taskId, progress: 0.0, message: 'processing tile' });
        try {
          post({ kind: 'progress', taskId: msg.taskId, progress: 0.2, message: 'preparing input' });
          const res = await runInferenceOnTile(msg.imageBuffer, msg.imageWidth, msg.imageHeight);
          if (!currentTasks.has(msg.taskId)) {
            post({ kind: 'progress', taskId: msg.taskId, progress: 1, message: 'cancelled' });
            break;
          }
          post({ kind: 'tileResult', taskId: msg.taskId, tile: msg.tile, imageBuffer: res.buffer, width: res.width, height: res.height }, [res.buffer]);
        } finally {
          currentTasks.delete(msg.taskId);
        }
        break;
      }
      case 'bypassUpscale': {
        // Receive full RGBA image and return a simple canvas-resized image (no model)
        try {
          const w = msg.width;
          const h = msg.height;
          const scale = msg.scale || 2;
          const src = new Uint8ClampedArray(msg.imageBuffer);
          const srcCanvas = new OffscreenCanvas(w, h);
          const sctx = srcCanvas.getContext('2d');
          const srcImg = new ImageData(src, w, h);
          sctx.putImageData(srcImg, 0, 0);
          const dstW = Math.floor(w * scale);
          const dstH = Math.floor(h * scale);
          const dstCanvas = new OffscreenCanvas(dstW, dstH);
          const dctx = dstCanvas.getContext('2d');
          dctx.drawImage(srcCanvas, 0, 0, dstW, dstH);
          const out = dctx.getImageData(0, 0, dstW, dstH).data;
          post({ kind: 'debug', taskId: msg.taskId, message: 'bypassUpscale returning resized image', width: dstW, height: dstH });
          post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: out.buffer, width: dstW, height: dstH }, [out.buffer]);
        } catch (e: any) {
          post({ kind: 'error', taskId: msg.taskId, message: 'bypassUpscale failed: ' + (e?.message || String(e)) });
        }
        break;
      }
      case 'processImage': {
        currentTasks.add(msg.taskId);
        post({ kind: 'progress', taskId: msg.taskId, progress: 0.2, message: 'Processing high-fidelity sub-pixel canvas...' });
        try {
          const srcW = msg.width;
          const srcH = msg.height;
          const targetScale = msg.scale || 2;
          let targetW = Math.floor(srcW * targetScale);
          let targetH = Math.floor(srcH * targetScale);

          if (targetScale === 8 || (msg.mode === 'document8k' && targetScale >= 4)) {
            const min8KWidth = 7680;
            if (targetW < min8KWidth) {
              targetW = min8KWidth;
              targetH = Math.round(targetW * (srcH / srcW));
            }
          }

          const srcU8 = new Uint8ClampedArray(msg.imageBuffer);
          const srcCanvas = new OffscreenCanvas(srcW, srcH);
          const sctx = srcCanvas.getContext('2d')!;
          sctx.putImageData(new ImageData(srcU8, srcW, srcH), 0, 0);

          const dstCanvas = new OffscreenCanvas(targetW, targetH);
          const dctx = dstCanvas.getContext('2d')!;
          dctx.imageSmoothingEnabled = true;
          dctx.imageSmoothingQuality = 'high';
          dctx.drawImage(srcCanvas, 0, 0, targetW, targetH);

          post({ kind: 'progress', taskId: msg.taskId, progress: 0.6, message: 'Applying enhancement filter...' });
          const resizedData = dctx.getImageData(0, 0, targetW, targetH).data;
          const sharpnessAmount = msg.sharpness !== undefined ? msg.sharpness : 0.3;
          const darknessAmount = msg.darkness !== undefined ? msg.darkness : 0.18;
          const roundnessAmount = msg.roundness !== undefined ? msg.roundness : 0.6;
          const isDoc = msg.mode === 'document8k';

          const finalU8 = apply8KVectorDocumentEngine(resizedData, targetW, targetH, sharpnessAmount, darknessAmount, roundnessAmount, isDoc);

          post({ kind: 'progress', taskId: msg.taskId, progress: 1.0, message: 'Complete' });
          post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: finalU8.buffer, width: targetW, height: targetH }, [finalU8.buffer]);
        } catch (e: any) {
          post({ kind: 'error', taskId: msg.taskId, message: 'Upscale failed: ' + String(e) });
        } finally {
          currentTasks.delete(msg.taskId);
        }
        break;
      }
      case 'cancel': {
        currentTasks.delete(msg.taskId);
        post({ kind: 'progress', taskId: msg.taskId, progress: 1, message: 'cancelled' });
        break;
      }
      default:
        post({ kind: 'error', message: 'unknown message kind' });
    }
  } catch (err: any) {
    post({ kind: 'error', taskId: msg?.taskId, message: err?.message || String(err) });
  }
};
