import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-onnx-wasm',
      configureServer(server) {
        server.middlewares.use('/ort-files', (req, res, next) => {
          const fileName = req.url ? req.url.replace(/^\//, '').split('?')[0] : ''
          const filePath = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist', fileName)
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            if (filePath.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm')
            if (filePath.endsWith('.mjs') || filePath.endsWith('.js')) res.setHeader('Content-Type', 'application/javascript; charset=utf-8')
            return fs.createReadStream(filePath).pipe(res)
          }
          next()
        })
      }
    }
  ],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2020',
  },
  publicDir: 'public',
})
