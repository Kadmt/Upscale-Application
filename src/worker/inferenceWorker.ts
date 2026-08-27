/* eslint-disable no-restricted-globals */
// WebWorker for running ONNXRuntime-Web inference (WebGPU preferred, WASM fallback)
import * as ort from 'onnxruntime-web'

// Configure WASM static asset paths for ONNX Runtime Web via absolute origin
if (ort?.env?.wasm) {
  const origin = typeof self !== 'undefined' && self.location && self.location.origin ? self.location.origin : '';
  ort.env.wasm.wasmPaths = origin + '/ort-files/';
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
        if (!session) {
          throw new Error('Session is not initialized yet. Please wait for the model to finish loading.');
        }
        post({ kind: 'progress', taskId: msg.taskId, progress: 0.0, message: 'tiling image' });
        // Simple single-shot implementation: treat the full image as one tile (not ideal for large images)
        try {
              // Debug: report session metadata and chosen strategy
              try {
                post({ kind: 'debug', taskId: msg.taskId, message: 'processImage start', sessionInputMetadata: session.inputMetadata, sessionOutputMetadata: session.outputMetadata });
              } catch (e) {}
          post({ kind: 'progress', taskId: msg.taskId, progress: 0.4, message: 'running inference' });
          // Check model input metadata for fixed tile size (e.g., [batch,1,224,224])
          const inputNames = session.inputNames || (session.inputMetadata ? Object.keys(session.inputMetadata) : ['input']);
          const inputMeta = resolveMetadataByName(session.inputMetadata, inputNames[0]);
          const expectedChannels = inputMeta && inputMeta.dims ? Number(inputMeta.dims[1]) || 1 : 1;
          const modelH = inputMeta && inputMeta.dims && Number(inputMeta.dims[2]) > 0 ? Number(inputMeta.dims[2]) : undefined;
          const modelW = inputMeta && inputMeta.dims && Number(inputMeta.dims[3]) > 0 ? Number(inputMeta.dims[3]) : undefined;
          const outMeta = resolveMetadataByName(session.outputMetadata, session.outputNames ? session.outputNames[0] : undefined);
          const outH = outMeta && outMeta.dims && Number(outMeta.dims[2]) > 0 ? Number(outMeta.dims[2]) : undefined;
          const outW = outMeta && outMeta.dims && Number(outMeta.dims[3]) > 0 ? Number(outMeta.dims[3]) : undefined;

          if (expectedChannels === 1 && modelH && modelW && outH && outW) {
            post({ kind: 'debug', taskId: msg.taskId, message: `using tiled strategy model=${modelW}x${modelH} out=${outW}x${outH} src=${msg.width}x${msg.height}` });
            // tiled processing: input tiles of modelH x modelW, output tiles of outH x outW
            const scaleY = outH / modelH;
            const scaleX = outW / modelW;
            const srcU8 = new Uint8ClampedArray(msg.imageBuffer);
            const srcW = msg.width;
            const srcH = msg.height;
            // prepare output Y buffer (float32) sized to srcW*scaleX by srcH*scaleY
            const dstW = Math.floor(srcW * scaleX);
            const dstH = Math.floor(srcH * scaleY);
            const outY = new Float32Array(dstW * dstH);

            // helper to extract RGBA tile from full image
            function extractTile(x: number, y: number, w: number, h: number) {
              const tile = new Uint8ClampedArray(w * h * 4);
              let p = 0;
              for (let j = 0; j < h; j++) {
                const sy = y + j;
                for (let i = 0; i < w; i++) {
                  const sx = x + i;
                  const sIdx = (sy * srcW + sx) * 4;
                  tile[p++] = srcU8[sIdx];
                  tile[p++] = srcU8[sIdx + 1];
                  tile[p++] = srcU8[sIdx + 2];
                  tile[p++] = srcU8[sIdx + 3];
                }
              }
              return tile;
            }

            // quick zero-tensor probe to verify model accepts [1,1,modelH,modelW]
            try {
              const probeShape = buildShapeFromMeta(inputMeta, 1, modelH, modelW);
              const probeSize = probeShape.reduce((a: number, b: number) => a * b, 1);
              const zeroProbe = new ort.Tensor('float32', new Float32Array(probeSize).fill(0), probeShape);
              const probeFeeds: AnyObject = {};
              const probeName = inputNames[0] || 'input';
              probeFeeds[probeName] = zeroProbe;
              await session.run(probeFeeds);
              post({ kind: 'progress', taskId: msg.taskId, message: `zero-probe succeeded for ${probeName} shape=${probeShape.join('x')}` });
            } catch (probeErr: any) {
              post({ kind: 'progress', taskId: msg.taskId, message: `zero-probe failed: ${probeErr?.message || String(probeErr)}` });
            }

            // overlapping tiled processing with reflection padding + simple averaging blending
            const overlap = Math.max(0, Math.floor(Math.min(modelW, modelH) / 4));
            const strideX = Math.max(1, modelW - overlap);
            const strideY = Math.max(1, modelH - overlap);

            // helper: reflect-pad RGBA tile into modelW x modelH
            function reflectPadRGBA(tile: Uint8ClampedArray, tw: number, th: number, dstW: number, dstH: number) {
              const out = new Uint8ClampedArray(dstW * dstH * 4);
              for (let y = 0; y < dstH; y++) {
                // source y with reflection
                let sy = y;
                if (sy < 0) sy = -sy - 1;
                if (sy >= th) sy = th - 1 - (sy - th);
                sy = Math.max(0, Math.min(th - 1, sy));
                for (let x = 0; x < dstW; x++) {
                  let sx = x;
                  if (sx < 0) sx = -sx - 1;
                  if (sx >= tw) sx = tw - 1 - (sx - tw);
                  sx = Math.max(0, Math.min(tw - 1, sx));
                  const sIdx = (sy * tw + sx) * 4;
                  const dIdx = (y * dstW + x) * 4;
                  out[dIdx] = tile[sIdx];
                  out[dIdx + 1] = tile[sIdx + 1];
                  out[dIdx + 2] = tile[sIdx + 2];
                  out[dIdx + 3] = tile[sIdx + 3] || 255;
                }
              }
              return out;
            }

            // accumulator arrays for Y and weights
            const accY = new Float32Array(dstW * dstH).fill(0);
            const accW = new Float32Array(dstW * dstH).fill(0);

            for (let ty = 0; ty < srcH; ty += strideY) {
              for (let tx = 0; tx < srcW; tx += strideX) {
                if (!currentTasks.has(msg.taskId)) break;
                const tw = Math.min(modelW, srcW - tx);
                const th = Math.min(modelH, srcH - ty);
                let tileU8 = extractTile(tx, ty, tw, th);
                if (tw !== modelW || th !== modelH) {
                  tileU8 = reflectPadRGBA(tileU8, tw, th, modelW, modelH);
                }
                const yFloat = uint8ToFloat32Y(tileU8);
                // debug: report input Y stats for this tile
                try {
                  let minIn = Infinity, maxIn = -Infinity, sumIn = 0;
                  for (let i = 0; i < yFloat.length; i++) {
                    const v = yFloat[i];
                    if (v < minIn) minIn = v;
                    if (v > maxIn) maxIn = v;
                    sumIn += v;
                  }
                  const meanIn = sumIn / yFloat.length;
                  const sampleIn = Array.from(yFloat.slice(0, 20));
                  post({ kind: 'debug', taskId: msg.taskId, message: `tile input stats tx=${tx} ty=${ty} tw=${tw} th=${th} min=${minIn.toFixed(4)} max=${maxIn.toFixed(4)} mean=${meanIn.toFixed(4)} sample=${JSON.stringify(sampleIn.slice(0,6))}`, tx, ty, tw, th, minIn, maxIn, meanIn, sampleIn });
                } catch (e) {}
                const tensorNCHW = makeInputTensor(yFloat, modelW, modelH, 1);
                try {
                  post({ kind: 'progress', taskId: msg.taskId, message: `running tile ${tx},${ty} size=${tw}x${th}` });
                  const feeds: AnyObject = {};
                  const inName = inputNames[0] || 'input';
                  feeds[inName] = tensorNCHW;
                  const outMap = await session.run(feeds);
                  const outName = session.outputNames ? session.outputNames[0] : Object.keys(outMap)[0];
                  const outTensor = outMap[outName];
                  const outData = outTensor.data as Float32Array; // outH*outW

                  // Per-tile post-process: normalize 0..255 range to 0..1 if needed and clamp
                  try {
                    let tMin = Infinity, tMax = -Infinity;
                    for (let ii = 0; ii < outData.length; ii++) {
                      const vv = outData[ii];
                      if (vv < tMin) tMin = vv;
                      if (vv > tMax) tMax = vv;
                    }
                    const isScale255 = tMax > 1.5;
                    for (let ii = 0; ii < outData.length; ii++) {
                      let vv = isScale255 ? outData[ii] / 255.0 : outData[ii];
                      if (!Number.isFinite(vv)) vv = 0;
                      if (vv < 0) vv = 0;
                      if (vv > 1) vv = 1;
                      outData[ii] = vv;
                    }
                  } catch (e) {}

                  // debug: report output stats for this tile
                  try {
                    let minOut = Infinity, maxOut = -Infinity, sumOut = 0;
                    for (let i = 0; i < outData.length; i++) {
                      const v = outData[i];
                      if (v < minOut) minOut = v;
                      if (v > maxOut) maxOut = v;
                      sumOut += v;
                    }
                    const meanOut = sumOut / outData.length;
                    const sampleOut = Array.from(outData.slice(0, 20));
                    post({ kind: 'debug', taskId: msg.taskId, message: `tile output stats tx=${tx} ty=${ty} out=${outW}x${outH} min=${minOut.toFixed(4)} max=${maxOut.toFixed(4)} mean=${meanOut.toFixed(4)} sample=${JSON.stringify(sampleOut.slice(0,6))}`, tx, ty, outW, outH, minOut, maxOut, meanOut, sampleOut });
                  } catch (e) {}

                  // compute destination top-left in output coords
                  const dstX = Math.round(tx * scaleX);
                  const dstY = Math.round(ty * scaleY);

                  for (let r = 0; r < outH; r++) {
                    const gy = dstY + r;
                    if (gy < 0 || gy >= dstH) continue;
                    for (let ccol = 0; ccol < outW; ccol++) {
                      const gx = dstX + ccol;
                      if (gx < 0 || gx >= dstW) continue;
                      const v = outData[r * outW + ccol];
                      const idx = gy * dstW + gx;
                      accY[idx] += v;
                      accW[idx] += 1;
                    }
                  }
                } catch (tileErr: any) {
                  post({ kind: 'progress', taskId: msg.taskId, message: `tile error ${tx},${ty}: ${tileErr?.message || String(tileErr)}` });
                  throw tileErr;
                }
              }
              if (!currentTasks.has(msg.taskId)) break;
            }

            // normalize accumulators into outY
            for (let i = 0; i < dstW * dstH; i++) {
              const w = accW[i];
              outY[i] = w > 0 ? (accY[i] / w) : 0;
            }

            // (debug Y return moved later to after normalization)

            // debug: inspect outY range to detect scaling issues (model may output 0..255 or -1..1)
            try {
              let minY = Infinity, maxY = -Infinity, sumY = 0;
              for (let i = 0; i < outY.length; i++) {
                const v = outY[i];
                if (v < minY) minY = v;
                if (v > maxY) maxY = v;
                sumY += v;
              }
              const meanY = sumY / outY.length;
              post({ kind: 'debug', taskId: msg.taskId, message: 'outY stats', minY, maxY, meanY });
              // auto-detect scale: if outputs look like 0..255, scale down; if -1..1, remap to 0..1
              let yScale = 1;
              let yOffset = 0;
              if (maxY > 1.5 && maxY > 50) {
                // assume 0..255
                yScale = 1 / 255;
                post({ kind: 'debug', taskId: msg.taskId, message: 'detected outY in 0..255, applying /255 scale' });
              } else if (minY < -0.5 && maxY <= 1.5) {
                // assume -1..1 -> map to 0..1
                yScale = 0.5;
                yOffset = 0.5;
                post({ kind: 'debug', taskId: msg.taskId, message: 'detected outY in -1..1, applying (v*0.5+0.5) remap' });
              }
              // Quick-fix: if outputs are slightly >1 (e.g. up to ~2) scale by maxY to avoid clipping to white
              if (yScale === 1 && maxY > 1.05 && maxY <= 50) {
                post({ kind: 'debug', taskId: msg.taskId, message: `detected outY max=${maxY.toFixed(4)} >1.05, applying global scale /maxY` });
                for (let i = 0; i < outY.length; i++) outY[i] = outY[i] / maxY;
              }
              if (yScale !== 1 || yOffset !== 0) {
                for (let i = 0; i < outY.length; i++) {
                  outY[i] = outY[i] * yScale + yOffset;
                }
              }
              // NOTE: disabled global contrast normalization — per-tile clamping/scaling used instead
            } catch (e) {
              // ignore
            }
              // If caller requested debug Y output, return grayscale Y image directly (post-processed)
              if ((msg as any).debugMode === 'Y') {
                try {
                  const outU8Gray = new Uint8ClampedArray(dstW * dstH * 4);
                  for (let i = 0; i < dstW * dstH; i++) {
                    const yv = Math.max(0, Math.min(1, outY[i]));
                    const v = Math.round(yv * 255);
                    outU8Gray[i * 4 + 0] = v;
                    outU8Gray[i * 4 + 1] = v;
                    outU8Gray[i * 4 + 2] = v;
                    outU8Gray[i * 4 + 3] = 255;
                  }
                  post({ kind: 'debug', taskId: msg.taskId, message: 'returning debug Y image' });
                  post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: outU8Gray.buffer, width: dstW, height: dstH }, [outU8Gray.buffer]);
                  currentTasks.delete(msg.taskId);
                  break;
                } catch (e) {
                  post({ kind: 'error', taskId: msg.taskId, message: 'failed to build debug Y image: ' + (e?.message || String(e)) });
                }
              }
            if (!currentTasks.has(msg.taskId)) {
              post({ kind: 'progress', taskId: msg.taskId, progress: 1, message: 'cancelled' });
              break;
            }

            // upsample original Cb/Cr to dstW x dstH using canvas
            const srcCanvas = new OffscreenCanvas(srcW, srcH);
            const sctx = srcCanvas.getContext('2d');
            const srcImage = new ImageData(new Uint8ClampedArray(srcU8), srcW, srcH);
            sctx.putImageData(srcImage, 0, 0);
            const dstCanvas = new OffscreenCanvas(dstW, dstH);
            const dctx = dstCanvas.getContext('2d');
            if (!dctx) {
              post({ kind: 'debug', taskId: msg.taskId, message: 'dstCanvas.getContext returned null' });
            } else {
              try {
                dctx.drawImage(srcCanvas, 0, 0, dstW, dstH);
              } catch (errDraw) {
                post({ kind: 'debug', taskId: msg.taskId, message: 'drawImage threw', error: String(errDraw) });
              }
            }
            let dstImg: Uint8ClampedArray;
            try {
              const im = dctx ? dctx.getImageData(0, 0, dstW, dstH).data : new Uint8ClampedArray(dstW * dstH * 4).fill(255);
              dstImg = im;
            } catch (errImg) {
              post({ kind: 'debug', taskId: msg.taskId, message: 'getImageData failed, using blank array', error: String(errImg) });
              dstImg = new Uint8ClampedArray(dstW * dstH * 4).fill(255);
            }

            // combine outY (float 0..1) with dstImg chroma derived from RGB of dstImg
            // Use ITU-R BT.601 (studio) formulas for conversion
            // Cb = (B - Yorig) / 1.772
            // Cr = (R - Yorig) / 1.402
            // R = Y + 1.402 * Cr
            // G = Y - 0.344136 * Cb - 0.714136 * Cr
            // B = Y + 1.772 * Cb
            const outU8 = new Uint8ClampedArray(dstW * dstH * 4);

            // detect blank/white dstImg (common if drawImage failed or returned blank canvas)
            let dstBlank = true;
            for (let k = 0; k < Math.min(12, dstImg.length); k++) {
              if (dstImg[k] !== 255) { dstBlank = false; break; }
            }
            if (dstBlank) {
              post({ kind: 'debug', taskId: msg.taskId, message: 'dstImg appears blank; using bilinear chroma fallback' });
            }

            // bilinear sampler for source image (srcU8)
            function bilinearSampleRGB(srcArr: Uint8ClampedArray, sW: number, sH: number, fx: number, fy: number) {
              // clamp coordinates
              if (fx <= 0) fx = 0;
              if (fy <= 0) fy = 0;
              if (fx >= sW - 1) fx = sW - 1;
              if (fy >= sH - 1) fy = sH - 1;
              const x0 = Math.floor(fx);
              const y0 = Math.floor(fy);
              const x1 = Math.min(sW - 1, x0 + 1);
              const y1 = Math.min(sH - 1, y0 + 1);
              const dx = fx - x0;
              const dy = fy - y0;
              const i00 = (y0 * sW + x0) * 4;
              const i10 = (y0 * sW + x1) * 4;
              const i01 = (y1 * sW + x0) * 4;
              const i11 = (y1 * sW + x1) * 4;
              const r00 = srcArr[i00] / 255; const g00 = srcArr[i00 + 1] / 255; const b00 = srcArr[i00 + 2] / 255;
              const r10 = srcArr[i10] / 255; const g10 = srcArr[i10 + 1] / 255; const b10 = srcArr[i10 + 2] / 255;
              const r01 = srcArr[i01] / 255; const g01 = srcArr[i01 + 1] / 255; const b01 = srcArr[i01 + 2] / 255;
              const r11 = srcArr[i11] / 255; const g11 = srcArr[i11 + 1] / 255; const b11 = srcArr[i11 + 2] / 255;
              const r0 = r00 * (1 - dx) + r10 * dx;
              const r1 = r01 * (1 - dx) + r11 * dx;
              const r = r0 * (1 - dy) + r1 * dy;
              const g0 = g00 * (1 - dx) + g10 * dx;
              const g1 = g01 * (1 - dx) + g11 * dx;
              const g = g0 * (1 - dy) + g1 * dy;
              const b0 = b00 * (1 - dx) + b10 * dx;
              const b1 = b01 * (1 - dx) + b11 * dx;
              const b = b0 * (1 - dy) + b1 * dy;
              return { r, g, b };
            }

            for (let i = 0; i < dstW * dstH; i++) {
              const yv = Math.max(0, Math.min(1, outY[i]));
              let r0: number, g0: number, b0: number;
              if (!dstBlank) {
                r0 = dstImg[i * 4 + 0] / 255;
                g0 = dstImg[i * 4 + 1] / 255;
                b0 = dstImg[i * 4 + 2] / 255;
              } else {
                // bilinear sample from original srcU8 mapped to dst coords
                const dx = i % dstW;
                const dy = Math.floor(i / dstW);
                // map dst pixel center to source float coords
                const srcX = (dx + 0.5) * (srcW / dstW) - 0.5;
                const srcY = (dy + 0.5) * (srcH / dstH) - 0.5;
                const col = bilinearSampleRGB(srcU8, srcW, srcH, srcX, srcY);
                r0 = col.r; g0 = col.g; b0 = col.b;
              }
              // compute original luma from upscaled (or sampled) RGB using BT.709
              const Yorig = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
              const Cb = (b0 - Yorig) / 1.8556;
              const Cr = (r0 - Yorig) / 1.5748;
              // reconstruct RGB using model Y (yv) and BT.709 inverse
              let R = yv + 1.5748 * Cr;
              let B = yv + 1.8556 * Cb;
              let G = (yv - 0.2126 * R - 0.0722 * B) / 0.7152;
              R = Math.max(0, Math.min(1, R));
              G = Math.max(0, Math.min(1, G));
              B = Math.max(0, Math.min(1, B));
              outU8[i * 4 + 0] = Math.round(R * 255);
              outU8[i * 4 + 1] = Math.round(G * 255);
              outU8[i * 4 + 2] = Math.round(B * 255);
              outU8[i * 4 + 3] = 255;
            }

            try {
              const sampleOutY = Array.from(outY.slice(0, 20));
              const sampleDst = Array.from(dstImg.slice(0, 12));
              const sampleRGBA = Array.from(outU8.slice(0, 40));
              const tailRGBA = Array.from(outU8.slice(Math.max(0, outU8.length - 40)));
              let minU = 255, maxU = 0, sumU = 0;
              for (let i = 0; i < outU8.length; i++) {
                const v = outU8[i];
                if (v < minU) minU = v;
                if (v > maxU) maxU = v;
                sumU += v;
              }
              const meanU = sumU / outU8.length;
              post({ kind: 'debug', taskId: msg.taskId, message: 'final tiled samples', sampleOutY, sampleDstLength: dstImg.length, sampleDstFirst12: sampleDst, sampleRGBAFirst40: sampleRGBA, sampleRGBALast40: tailRGBA, minU, maxU, meanU });
            } catch (e) {
              post({ kind: 'debug', taskId: msg.taskId, message: 'failed to build final tiled samples', error: String(e) });
            }

            // High-Fidelity 8K Vector-Smooth Document Restoration Engine
            function apply8KVectorDocumentEngine(
              rgba: Uint8ClampedArray,
              w: number,
              h: number,
              strength = 0.3,
              darknessFactor = 0.18,
              isDocument8K = true
            ) {
              if (strength <= 0 && darknessFactor <= 0 && !isDocument8K) return rgba;
              const out = new Uint8ClampedArray(rgba.length);

              // Calculate document background luma estimate
              let bgSum = 0;
              let bgCnt = 0;
              for (let i = 0; i < rgba.length; i += 16) {
                const l = 0.2126 * rgba[i] + 0.7152 * rgba[i + 1] + 0.0722 * rgba[i + 2];
                if (l > 140) {
                  bgSum += l;
                  bgCnt++;
                }
              }
              const paperBg = bgCnt > 0 ? bgSum / bgCnt : 215;

              for (let y = 0; y < h; y++) {
                const y0 = Math.max(0, y - 1) * w;
                const y1 = Math.min(h - 1, y + 1) * w;
                const yw = y * w;

                for (let x = 0; x < w; x++) {
                  const x0 = Math.max(0, x - 1);
                  const x1 = Math.min(w - 1, x + 1);
                  const idx = (yw + x) * 4;

                  const iTC = (y0 + x) * 4;
                  const iCL = (yw + x0) * 4; const iCR = (yw + x1) * 4;
                  const iBC = (y1 + x) * 4;

                  for (let c = 0; c < 3; c++) {
                    const center = rgba[idx + c];
                    
                    // 1. Unsharp Mask for edge definition (preserving dots, periods & accents)
                    let val = center;
                    if (strength > 0) {
                      const blur = (rgba[iTC + c] + rgba[iCL + c] + rgba[iCR + c] + rgba[iBC + c] + 4 * center) / 8;
                      const diff = center - blur;
                      val = center + strength * 0.45 * diff;
                    }

                    if (isDocument8K) {
                      // 2. High-Contrast Pure White Background Whitening (#FFFFFF)
                      if (val > paperBg - 25) {
                        const factor = Math.min(1.0, (val - (paperBg - 25)) / Math.max(1, 255 - (paperBg - 25)));
                        val = val + (255 - val) * Math.pow(factor, 0.5);
                      } else if (val < 140) {
                        // 3. Solid Deep Black Ink Solidification (#000000)
                        const boost = (1.0 - (val / 140.0) * (val / 140.0)) * Math.max(darknessFactor, 0.22);
                        val = val * (1.0 - boost);
                      }
                    } else if (darknessFactor > 0 && val < 175) {
                      const t = val / 175.0;
                      const boost = (1.0 - t * t) * darknessFactor;
                      val = val * (1.0 - boost);
                    }

                    out[idx + c] = Math.max(0, Math.min(255, Math.round(val)));
                  }
                  out[idx + 3] = rgba[idx + 3];
                }
              }
              return out;
            }

            const sharpnessAmount = msg.sharpness !== undefined ? msg.sharpness : 0.3;
            const darknessAmount = msg.darkness !== undefined ? msg.darkness : 0.18;
            const isDoc = msg.mode === 'document8k' || msg.mode === undefined || msg.mode === 'text';
            const sharpenedOutU8 = apply8KVectorDocumentEngine(outU8, dstW, dstH, sharpnessAmount, darknessAmount, isDoc);

            // Resample to target user scale (2x, 3x, 4x, 8K Ultra)
            const targetScale = msg.scale || 2;
            let targetW = Math.floor(srcW * targetScale);
            let targetH = Math.floor(srcH * targetScale);

            // True 8K Output Target Option (7680px Width Standard)
            if (targetScale === 8 || (msg.mode === 'document8k' && targetScale >= 4)) {
              const min8KWidth = 7680;
              if (targetW < min8KWidth) {
                targetW = min8KWidth;
                targetH = Math.round(targetW * (srcH / srcW));
              }
            }

            let finalBuffer: ArrayBuffer = sharpenedOutU8.buffer;
            let finalW = dstW;
            let finalH = dstH;

            if (dstW !== targetW || dstH !== targetH) {
              const srcCanvas = new OffscreenCanvas(dstW, dstH);
              const sctx = srcCanvas.getContext('2d')!;
              sctx.putImageData(new ImageData(sharpenedOutU8, dstW, dstH), 0, 0);

              const dstCanvas = new OffscreenCanvas(targetW, targetH);
              const dctx = dstCanvas.getContext('2d')!;
              dctx.imageSmoothingEnabled = true;
              dctx.imageSmoothingQuality = 'high';
              dctx.drawImage(srcCanvas, 0, 0, targetW, targetH);

              const resizedData = dctx.getImageData(0, 0, targetW, targetH).data;

              // Apply High-Density 8K Sub-Pixel Vector Refinement (Fast Uint32 Buffer View)
              if (targetW >= 3840 && isDoc) {
                const u32 = new Uint32Array(resizedData.buffer);
                const len = u32.length;
                for (let i = 0; i < len; i++) {
                  const pixel = u32[i];
                  const r = pixel & 0xff;
                  const g = (pixel >> 8) & 0xff;
                  const b = (pixel >> 16) & 0xff;
                  const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                  if (luma > 215) {
                    u32[i] = 0xffffffff;
                  } else if (luma < 95) {
                    u32[i] = 0xff050505;
                  }
                }
              }

              finalBuffer = resizedData.buffer;
              finalW = targetW;
              finalH = targetH;
            }

            post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: finalBuffer, width: finalW, height: finalH }, [finalBuffer]);
          } else {
            post({ kind: 'debug', taskId: msg.taskId, message: `using single-shot fallback src=${msg.width}x${msg.height} expectedChannels=${expectedChannels}` });
            // fallback: single-shot
            const res = await runInferenceOnTile(msg.imageBuffer, msg.width, msg.height);
            if (!currentTasks.has(msg.taskId)) {
              post({ kind: 'progress', taskId: msg.taskId, progress: 1, message: 'cancelled' });
              break;
            }
            post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: res.buffer, width: res.width, height: res.height }, [res.buffer]);
          }
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
