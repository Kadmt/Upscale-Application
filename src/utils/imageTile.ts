// Utilities for extracting tiles and blending tiles into a destination buffer

export function extractTileRGBA(
  src: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  x: number,
  y: number,
  w: number,
  h: number
) {
  const out = new Uint8ClampedArray(w * h * 4);
  let p = 0;
  for (let j = 0; j < h; j++) {
    const sy = y + j;
    for (let i = 0; i < w; i++) {
      const sx = x + i;
      if (sx < 0 || sx >= srcW || sy < 0 || sy >= srcH) {
        out[p++] = 0; out[p++] = 0; out[p++] = 0; out[p++] = 0;
      } else {
        const sIdx = (sy * srcW + sx) * 4;
        out[p++] = src[sIdx];
        out[p++] = src[sIdx + 1];
        out[p++] = src[sIdx + 2];
        out[p++] = src[sIdx + 3];
      }
    }
  }
  return out;
}

export function blendTileInto(
  dest: Uint8ClampedArray,
  destW: number,
  destH: number,
  tile: Uint8ClampedArray,
  x: number,
  y: number,
  tileW: number,
  tileH: number,
  overlap: number = 0
) {
  // Simple overwrite blend with optional naive feathering on edges
  for (let j = 0; j < tileH; j++) {
    const dy = y + j;
    if (dy < 0 || dy >= destH) continue;
    for (let i = 0; i < tileW; i++) {
      const dx = x + i;
      if (dx < 0 || dx >= destW) continue;
      const tIdx = (j * tileW + i) * 4;
      const dIdx = (dy * destW + dx) * 4;
      // naive: full overwrite
      dest[dIdx] = tile[tIdx];
      dest[dIdx + 1] = tile[tIdx + 1];
      dest[dIdx + 2] = tile[tIdx + 2];
      dest[dIdx + 3] = tile[tIdx + 3];
    }
  }
}
