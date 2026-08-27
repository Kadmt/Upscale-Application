const fs = require('fs')
const path = require('path')

async function download(url, outPath) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  const arrayBuffer = await res.arrayBuffer()
  const buffer = Buffer.from(arrayBuffer)
  await fs.promises.mkdir(path.dirname(outPath), { recursive: true })
  await fs.promises.writeFile(outPath, buffer)
}

async function main() {
  const url = process.argv[2] || process.env.MODEL_URL || 'https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super_resolution.onnx'
  const out = process.argv[3] || process.env.OUT_PATH || path.join(__dirname, '..', 'public', 'models', 'sample.onnx')
  console.log('Downloading', url)
  try {
    await download(url, out)
    console.log('Saved model to', out)
  } catch (err) {
    console.error('Download failed:', err.message)
    process.exit(1)
  }
}

main()
