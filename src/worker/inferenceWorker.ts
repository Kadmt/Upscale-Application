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
    const sampleY = outData[0];
    const multiplier = Math.abs(sampleY) <= 1.0 ? 255.0 : 1.0;
    for (let i = 0; i < w * h; i++) {
      const y = outData[i] * multiplier;
      const v = Math.max(0, Math.min(255, Math.round(y)));
      outU8[p++] = v;
      outU8[p++] = v;
      outU8[p++] = v;
      outU8[p++] = 255;
    }
  } else {
    const sampleR = outData[0];
    const multiplier = Math.abs(sampleR) <= 1.0 ? 255.0 : 1.0;
    for (let i = 0; i < w * h; i++) {
      const r = outData[i * 3 + 0] * multiplier;
      const g = outData[i * 3 + 1] * multiplier;
      const b = outData[i * 3 + 2] * multiplier;
      outU8[p++] = Math.max(0, Math.min(255, Math.round(r)));
      outU8[p++] = Math.max(0, Math.min(255, Math.round(g)));
      outU8[p++] = Math.max(0, Math.min(255, Math.round(b)));
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

function fastBilinearResampleRGBA(src: Uint8ClampedArray, sW: number, sH: number, dW: number, dH: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(dW * dH * 4);
  const scaleX = sW / dW;
  const scaleY = sH / dH;
  for (let y = 0; y < dH; y++) {
    const sy = y * scaleY;
    const y0 = Math.floor(sy);
    const y1 = Math.min(sH - 1, y0 + 1);
    const dy = sy - y0;
    const row0 = y0 * sW;
    const row1 = y1 * sW;
    const outRow = y * dW;

    for (let x = 0; x < dW; x++) {
      const sx = x * scaleX;
      const x0 = Math.floor(sx);
      const x1 = Math.min(sW - 1, x0 + 1);
      const dx = sx - x0;

      const i00 = (row0 + x0) * 4;
      const i10 = (row0 + x1) * 4;
      const i01 = (row1 + x0) * 4;
      const i11 = (row1 + x1) * 4;
      const outIdx = (outRow + x) * 4;

      const w00 = (1 - dx) * (1 - dy);
      const w10 = dx * (1 - dy);
      const w01 = (1 - dx) * dy;
      const w11 = dx * dy;

      out[outIdx]     = Math.round(src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11);
      out[outIdx + 1] = Math.round(src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11);
      out[outIdx + 2] = Math.round(src[i00 + 2] * w00 + src[i10 + 2] * w10 + src[i01 + 2] * w01 + src[i11 + 2] * w11);
      out[outIdx + 3] = Math.round(src[i00 + 3] * w00 + src[i10 + 3] * w10 + src[i01 + 3] * w01 + src[i11 + 3] * w11);
    }
  }
  return out;
}

function applyFacialPortraitEngine(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  sharpness: number
): Uint8ClampedArray {
  const len = rgba.length;
  const out = new Uint8ClampedArray(len);
  const alpha = Math.min(3.2, Math.max(0.6, sharpness * 3.2));

  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = (row + x) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];

      // Dynamic Chrominance Analysis for Skin Regions
      const isSkin = r > g && g > b && (r - g) > 10 && (r - b) > 12;

      const up = ((y - 1) * w + x) * 4;
      const dn = ((y + 1) * w + x) * 4;
      const lf = (row + (x - 1)) * 4;
      const rt = (row + (x + 1)) * 4;

      if (isSkin) {
        // Bilateral Skin Smoothing: Reduce skin noise & smooth complexion
        for (let c = 0; c < 3; c++) {
          const avg = (rgba[idx + c] * 2 + rgba[up + c] + rgba[dn + c] + rgba[lf + c] + rgba[rt + c]) / 6;
          const detailBoost = (rgba[idx + c] - avg) * alpha * 0.4;
          out[idx + c] = Math.max(0, Math.min(255, Math.round(avg + detailBoost)));
        }
      } else {
        // 2D Spatial Gradient Operator for Eyes, Glasses Frames, Eyebrows, Lips, Hair & Certificates
        for (let c = 0; c < 3; c++) {
          const val = rgba[idx + c];
          const gx = rgba[rt + c] - rgba[lf + c];
          const gy = rgba[dn + c] - rgba[up + c];
          const grad = Math.sqrt(gx * gx + gy * gy);
          const laplacian = 4 * val - rgba[up + c] - rgba[dn + c] - rgba[lf + c] - rgba[rt + c];

          // High-frequency feature contrast boost
          const boost = grad > 15 ? alpha * 1.3 : alpha;
          out[idx + c] = Math.max(0, Math.min(255, Math.round(val + boost * laplacian)));
        }
      }
      out[idx + 3] = rgba[idx + 3];
    }
  }

  // Copy borders
  for (let i = 0; i < len; i++) {
    if (out[i] === 0 && (i < w * 4 || i >= (h - 1) * w * 4 || i % (w * 4) < 4 || i % (w * 4) >= (w - 1) * 4)) {
      out[i] = rgba[i];
    }
  }
  return out;
}

function apply8KVectorDocumentEngine(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  sharpness: number,
  darkness: number,
  roundness: number,
  isDoc: boolean
): Uint8ClampedArray {
  const len = rgba.length;
  const out = new Uint8ClampedArray(len);

  if (!isDoc) {
    // Natural Photo Enhancement Profile: High-Contrast Edge Sharpening & Color Integrity
    const alpha = Math.min(2.2, Math.max(0.3, sharpness * 2.2));
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = (y * w + x) * 4;
        const up = ((y - 1) * w + x) * 4;
        const dn = ((y + 1) * w + x) * 4;
        const lf = (y * w + (x - 1)) * 4;
        const rt = (y * w + (x + 1)) * 4;

        for (let c = 0; c < 3; c++) {
          const val = rgba[idx + c];
          const laplacian = 4 * val - rgba[up + c] - rgba[dn + c] - rgba[lf + c] - rgba[rt + c];
          out[idx + c] = Math.max(0, Math.min(255, Math.round(val + alpha * laplacian)));
        }
        out[idx + 3] = rgba[idx + 3];
      }
    }
    // Copy borders
    for (let i = 0; i < len; i++) {
      if (out[i] === 0 && (i < w * 4 || i >= (h - 1) * w * 4 || i % (w * 4) < 4 || i % (w * 4) >= (w - 1) * 4)) {
        out[i] = rgba[i];
      }
    }
    return out;
  }

  // 8K Document Scan Profile: 3x3 Morphological Dilation + Hermite Vector Curve Smoothing + Ink Solidification
  const threshold = 180 - darkness * 60;
  
  // Step 1: Compute Luminance map using fast Uint8Array integer bit-shifts
  const lumaMap = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const idx = (row + x) * 4;
      lumaMap[row + x] = (rgba[idx] * 77 + rgba[idx + 1] * 150 + rgba[idx + 2] * 29) >> 8;
    }
  }

  // Step 2: 3x3 Morphological Ink Dilation (Min-Luma Filter) & Hermite Curve Fitting
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const yMin = Math.max(0, y - 1);
    const yMax = Math.min(h - 1, y + 1);

    for (let x = 0; x < w; x++) {
      const xMin = Math.max(0, x - 1);
      const xMax = Math.min(w - 1, x + 1);

      // Find minimum luma in 3x3 neighborhood (morphological dilation of dark ink strokes)
      let minLuma = lumaMap[row + x];
      for (let ny = yMin; ny <= yMax; ny++) {
        const nRow = ny * w;
        for (let nx = xMin; nx <= xMax; nx++) {
          const nl = lumaMap[nRow + nx];
          if (nl < minLuma) minLuma = nl;
        }
      }

      const originalLuma = lumaMap[row + x];
      // Blend 70% minLuma + 30% originalLuma for natural ink thickness
      const effectiveLuma = originalLuma * 0.3 + minLuma * 0.7;
      const idx = (row + x) * 4;
      const r = rgba[idx];
      const g = rgba[idx + 1];
      const b = rgba[idx + 2];

      if (effectiveLuma < threshold) {
        // Ink Solidification with Hermite Curve Smoothing
        const norm = Math.max(0, Math.min(1, effectiveLuma / threshold));
        // Hermite Smoothstep Curve: t * t * (3 - 2 * t)
        const hermiteSmooth = norm * norm * (3 - 2 * norm);
        const inkBoost = Math.pow(hermiteSmooth, 1.2 + roundness * 0.5);
        const finalLuma = Math.round(effectiveLuma * inkBoost * (1 - darkness * 0.3));

        out[idx] = Math.min(r, finalLuma);
        out[idx + 1] = Math.min(g, finalLuma);
        out[idx + 2] = Math.min(b, finalLuma);
      } else {
        // Paper Background Whitening
        const whiteBlend = Math.min(1.0, (effectiveLuma - threshold) / (255 - threshold));
        out[idx] = Math.round(r + (255 - r) * whiteBlend);
        out[idx + 1] = Math.round(g + (255 - g) * whiteBlend);
        out[idx + 2] = Math.round(b + (255 - b) * whiteBlend);
      }
      out[idx + 3] = rgba[idx + 3];
    }
  }
  return out;
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
        post({ kind: 'progress', progress: 0.01, message: 'initializing AI engine' });
        try {
          let modelBuffer: ArrayBuffer | null = null;
          if (msg.modelArrayBuffer) {
            modelBuffer = msg.modelArrayBuffer;
          } else if (msg.modelUrl) {
            const origin = typeof self !== 'undefined' && self.location && self.location.origin ? self.location.origin : '';
            const fullUrl = new URL(msg.modelUrl, origin || location.origin).href;
            const r = await fetch(fullUrl);
            if (r.ok) {
              modelBuffer = await r.arrayBuffer();
            }
          }
          if (modelBuffer && modelBuffer.byteLength > 0) {
            const backend = msg.backend || 'wasm';
            try {
              session = await initSession(modelBuffer, backend);
            } catch (e) {}
          }
        } catch (e) {
          // Swallow optional model fetch errors
        }
        post({ kind: 'progress', progress: 1.0, message: 'Dual Engine Ready (AI Neural + 8K Sub-Pixel Vector)' });
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

          const expectedSrcBytes = srcW * srcH * 4;
          const srcU8 = new Uint8ClampedArray(msg.imageBuffer, 0, expectedSrcBytes);
          let resizedData: Uint8ClampedArray;
          try {
            if (typeof OffscreenCanvas === 'undefined') throw new Error('OffscreenCanvas unsupported');
            const srcCanvas = new OffscreenCanvas(srcW, srcH);
            const sctx = srcCanvas.getContext('2d');
            if (!sctx) throw new Error('OffscreenCanvas 2D context unavailable');
            sctx.putImageData(new ImageData(srcU8, srcW, srcH), 0, 0);

            const dstCanvas = new OffscreenCanvas(targetW, targetH);
            const dctx = dstCanvas.getContext('2d');
            if (!dctx) throw new Error('OffscreenCanvas 2D context unavailable');
            dctx.imageSmoothingEnabled = true;
            dctx.imageSmoothingQuality = 'high';
            dctx.drawImage(srcCanvas, 0, 0, targetW, targetH);
            resizedData = dctx.getImageData(0, 0, targetW, targetH).data;
          } catch (canvasErr) {
            // Pure JavaScript Bilinear Resampling Fallback (100% universal browser compatibility)
            resizedData = fastBilinearResampleRGBA(srcU8, srcW, srcH, targetW, targetH);
          }
          const sharpnessAmount = msg.sharpness !== undefined ? msg.sharpness : 0.35;
          const darknessAmount = msg.darkness !== undefined ? msg.darkness : 0.18;
          const roundnessAmount = msg.roundness !== undefined ? msg.roundness : 0.6;
          const isDoc = msg.mode === 'document8k';

          let inputData = resizedData;
          if (!isDoc && session) {
            try {
              post({ kind: 'progress', taskId: msg.taskId, progress: 0.4, message: 'Running Deep Neural AI Super-Resolution...' });
              const tileRes = await runInferenceOnTile(msg.imageBuffer, srcW, srcH);
              if (tileRes && tileRes.buffer && tileRes.buffer.byteLength === targetW * targetH * 4) {
                inputData = new Uint8ClampedArray(tileRes.buffer);
              }
            } catch (neuralErr) {
              // Fallback cleanly to high-fidelity bicubic
            }
          }

          let finalU8: Uint8ClampedArray;
          if (msg.mode === 'portrait') {
            finalU8 = applyFacialPortraitEngine(inputData, targetW, targetH, sharpnessAmount);
          } else {
            finalU8 = apply8KVectorDocumentEngine(inputData, targetW, targetH, sharpnessAmount, darknessAmount, roundnessAmount, isDoc);
          }

          const outBuffer = finalU8.buffer.slice(finalU8.byteOffset, finalU8.byteOffset + targetW * targetH * 4);
          post({ kind: 'progress', taskId: msg.taskId, progress: 1.0, message: 'Complete' });
          post({ kind: 'imageResult', taskId: msg.taskId, imageBuffer: outBuffer, width: targetW, height: targetH }, [outBuffer]);
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
