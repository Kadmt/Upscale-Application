import React, { useRef, useState, useEffect } from 'react'
import Uploader from './components/Uploader'
import Preview from './components/Preview'
import ProcessingQueue from './components/ProcessingQueue'
import Settings from './components/Settings'
import { InferenceController } from './worker/controller'
import InferenceWorker from './worker/inferenceWorker?worker'

export default function App() {
  const [imageData, setImageData] = useState<ImageData | null>(null)
  const [processedImage, setProcessedImage] = useState<ImageData | null>(null)
  const [status, setStatus] = useState<string>('idle')
  const [progress, setProgress] = useState<number>(0)
  const [progressMsg, setProgressMsg] = useState<string>('')
  const [selectedScale, setSelectedScale] = useState<number>(2)
  const [sharpness, setSharpness] = useState<number>(0)
  const [darkness, setDarkness] = useState<number>(18)
  const [mode, setMode] = useState<'general' | 'document8k'>('document8k')

  const controllerRef = useRef<InferenceController | null>(null)

  useEffect(() => {
    const worker = new InferenceWorker()
    controllerRef.current = new InferenceController(worker, (p, m) => {
      if (typeof p === 'number' && !isNaN(p)) {
        setProgress(Math.round(p * 100))
      }
      if (m) setProgressMsg(m)
    })

    ;(window as any).__inferenceController = controllerRef.current
    controllerRef.current.init('/models/super-resolution-10.onnx')

    return () => controllerRef.current?.dispose()
  }, [])

  useEffect(() => {
    const onImage = (e: Event) => {
      const detail = (e as CustomEvent).detail
      try {
        const buf = detail.imageBuffer as ArrayBuffer
        const width = detail.width as number
        const height = detail.height as number

        if (!buf || buf.byteLength !== width * height * 4) {
          console.error('Invalid image buffer size — aborting render')
          setStatus('error')
          return
        }

        const u8 = new Uint8ClampedArray(buf)
        const img = new ImageData(u8, width, height)

        setProcessedImage(img)
        setStatus('done')
        setProgress(100)
        setProgressMsg('Neural upscale completed successfully!')
      } catch (err) {
        console.error('Failed to build image from result', err)
        setStatus('error')
      }
    }

    const onError = (e: Event) => {
      const detail = (e as CustomEvent).detail
      console.error('Inference error:', detail)
      if (detail && detail.taskId) {
        setStatus('error')
        setProgressMsg(detail?.message || 'Error occurred during neural inference')
      }
    }

    window.addEventListener('inference:image', onImage as any)
    window.addEventListener('inference:error', onError as any)

    return () => {
      window.removeEventListener('inference:image', onImage as any)
      window.removeEventListener('inference:error', onError as any)
    }
  }, [])

  function handleStart() {
    if (!imageData) {
      alert('Please select an image to upscale first.')
      return
    }
    setStatus('processing')
    setProgress(0)
    setProgressMsg('Processing neural upscaling tiles...')
    const id = `manual-${Date.now()}`
    const sharpnessStrength = sharpness / 100
    const darknessFactor = darkness / 100
    const optimalRoundness = 0.6
    controllerRef.current?.processFullImage(id, imageData, selectedScale, sharpnessStrength, darknessFactor, optimalRoundness, mode)
  }

  return (
    <div className="app">
      <header>
        <div className="header-brand">
          <div className="brand-icon">✨</div>
          <div>
            <h1>AI Document & Image Upscaler Studio</h1>
          </div>
        </div>
        <div className="header-badge">
          <span className="status-dot"></span>
          WASM SIMD Engine Ready
        </div>
      </header>

      <main>
        <section className="left">
          <Uploader onImage={(img) => {
            setImageData(img)
            setProcessedImage(null)
            setStatus('idle')
          }} />

          <Settings
            scale={selectedScale}
            sharpness={sharpness}
            darkness={darkness}
            mode={mode}
            onScaleChange={setSelectedScale}
            onSharpnessChange={setSharpness}
            onDarknessChange={setDarkness}
            onModeChange={setMode}
          />

          <div className="card action-card">
            <button className="btn-primary start-btn" onClick={handleStart} disabled={!imageData || status === 'processing'}>
              {status === 'processing' ? '⚡ Processing 8K Neural Tiles...' : '✨ Start AI Upscale'}
            </button>
          </div>
        </section>

        <section className="center">
          <Preview originalImage={imageData} processedImage={processedImage} />
        </section>

        <section className="right">
          <ProcessingQueue status={status} progress={progress} progressMessage={progressMsg} />
        </section>
      </main>
    </div>
  )
}
