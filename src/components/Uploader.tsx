import React, { useRef, useState, useEffect } from 'react'

export default function Uploader({ onImage }: { onImage: (img: ImageData) => void }) {
  const ref = useRef<HTMLInputElement | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  async function processFile(f: File) {
    setFileName(f.name)
    const img = await createImageBitmap(f)
    const c = new OffscreenCanvas(img.width, img.height)
    const ctx = c.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const imageData = ctx.getImageData(0, 0, img.width, img.height)
    try {
      ;(window as any).__lastUploadedImageData = imageData
    } catch (e) {}
    onImage(imageData)
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) processFile(f)
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f) processFile(f)
  }

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.indexOf('image') !== -1) {
          const blob = item.getAsFile()
          if (blob) processFile(blob)
          break
        }
      }
    }
    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [])

  return (
    <div className="card uploader-card">
      <div className="card-header">
        <h3>📁 Input Source Image</h3>
      </div>
      <div
        className="uploader-dropzone"
        onClick={() => ref.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input ref={ref} type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
        <div className="upload-icon">📸</div>
        <div className="upload-text">
          {fileName ? `Loaded: ${fileName}` : 'Click, Drag & Drop or Paste (Ctrl+V)'}
        </div>
        <div className="upload-subtext">Supports PNG, JPG, WEBP formats</div>
      </div>
    </div>
  )
}
