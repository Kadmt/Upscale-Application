const fs = require('fs')
const path = require('path')

async function copyWasmFiles() {
  const srcDir = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist')
  const destDir = path.join(__dirname, '..', 'public', 'ort-files')

  if (!fs.existsSync(srcDir)) {
    console.warn('onnxruntime-web dist folder not found at', srcDir)
    return
  }

  await fs.promises.mkdir(destDir, { recursive: true })
  const files = await fs.promises.readdir(srcDir)
  const wasmFiles = files.filter(f => f.endsWith('.wasm') || f.endsWith('.mjs'))

  let count = 0
  for (const file of wasmFiles) {
    const src = path.join(srcDir, file)
    const dest = path.join(destDir, file)
    await fs.promises.copyFile(src, dest)
    count++
  }
  console.log(`Successfully copied ${count} WASM/MJS files to ${destDir}`)
}

copyWasmFiles().catch(err => {
  console.error('Failed to copy WASM files:', err)
  process.exit(1)
})
