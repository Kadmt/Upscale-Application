import React, { useRef, useEffect, useState } from 'react'

interface PreviewProps {
  originalImage: ImageData | null
  processedImage: ImageData | null
}

export default function Preview({ originalImage, processedImage }: PreviewProps) {
  const procCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const origCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [zoom, setZoom] = useState<'fit' | '100' | '200'>('fit')
  const [splitPos, setSplitPos] = useState<number>(50) // 0 to 100%
  const [isDragging, setIsDragging] = useState<boolean>(false)

  const activeImage = processedImage || originalImage

  // Render Processed HD Canvas (Base Layer) ONCE when processedImage changes
  useEffect(() => {
    const canvas = procCanvasRef.current
    if (!canvas || !processedImage) return

    const w = processedImage.width
    const h = processedImage.height
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(processedImage, 0, 0)
  }, [processedImage])

  // Render Original Canvas (Top Layer) ONCE when originalImage/processedImage changes
  useEffect(() => {
    const canvas = origCanvasRef.current
    if (!canvas || !originalImage) return

    const targetW = processedImage ? processedImage.width : originalImage.width
    const targetH = processedImage ? processedImage.height : originalImage.height

    canvas.width = targetW
    canvas.height = targetH

    const ctx = canvas.getContext('2d')!
    const tmp = document.createElement('canvas')
    tmp.width = originalImage.width
    tmp.height = originalImage.height
    tmp.getContext('2d')!.putImageData(originalImage, 0, 0)

    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(tmp, 0, 0, targetW, targetH)
  }, [originalImage, processedImage])

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const pos = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSplitPos(pos)
  }

  const getZoomScale = () => {
    switch (zoom) {
      case '100': return 'scale(1)'
      case '200': return 'scale(2)'
      case 'fit':
      default: return 'none'
    }
  }

  const handleDownload = () => {
    if (!processedImage) return
    const canvas = document.createElement('canvas')
    canvas.width = processedImage.width
    canvas.height = processedImage.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(processedImage, 0, 0)

    const link = document.createElement('a')
    link.download = `upscaled-hd-${processedImage.width}x${processedImage.height}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }

  const handleDownloadComparison = () => {
    if (!originalImage || !processedImage) return
    const canvas = document.createElement('canvas')
    const w = processedImage.width
    const h = processedImage.height
    canvas.width = w * 2 + 20
    canvas.height = h + 40
    const ctx = canvas.getContext('2d')!

    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    const origCanvas = document.createElement('canvas')
    origCanvas.width = originalImage.width
    origCanvas.height = originalImage.height
    origCanvas.getContext('2d')!.putImageData(originalImage, 0, 0)
    ctx.drawImage(origCanvas, 0, 30, w, h)

    const procCanvas = document.createElement('canvas')
    procCanvas.width = w
    procCanvas.height = h
    procCanvas.getContext('2d')!.putImageData(processedImage, 0, 0)
    ctx.drawImage(procCanvas, w + 20, 30, w, h)

    ctx.fillStyle = '#0f172a'
    ctx.font = 'bold 16px sans-serif'
    ctx.fillText('Original Image', 10, 22)
    ctx.fillText('Upscaled HD Image', w + 30, 22)

    const link = document.createElement('a')
    link.download = `comparison-side-by-side-${Date.now()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }

  return (
    <div className="preview-workspace">
      <div className="preview-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="preview-title">🖼️ Studio Canvas View</span>
          {activeImage && (
            <span className="dimension-badge">
              {processedImage ? `Upscaled: ${activeImage.width} × ${activeImage.height}` : `Original: ${activeImage.width} × ${activeImage.height}`}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {processedImage && (
            <>
              <button
                className="btn-primary"
                style={{ fontSize: 12, padding: '5px 12px', background: '#059669' }}
                onClick={handleDownload}
              >
                📥 Download HD PNG
              </button>
              {originalImage && (
                <button
                  className="btn-secondary"
                  style={{ fontSize: 12, padding: '5px 10px' }}
                  onClick={handleDownloadComparison}
                  title="Download Side-by-Side Comparison Image"
                >
                  📊 Download Split View
                </button>
              )}
            </>
          )}
          <div style={{ width: 1, height: 20, background: '#cbd5e1', margin: '0 4px' }} />
          <button
            className="btn-secondary"
            style={{ background: zoom === 'fit' ? '#e2e8f0' : undefined }}
            onClick={() => setZoom('fit')}
          >
            Fit
          </button>
          <button
            className="btn-secondary"
            style={{ background: zoom === '100' ? '#e2e8f0' : undefined }}
            onClick={() => setZoom('100')}
          >
            100%
          </button>
          <button
            className="btn-secondary"
            style={{ background: zoom === '200' ? '#e2e8f0' : undefined }}
            onClick={() => setZoom('200')}
          >
            200%
          </button>
        </div>
      </div>

      <div
        className="canvas-viewport"
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={handleMouseMove}
      >
        {!activeImage ? (
          <div style={{ color: '#94a3b8', textAlign: 'center', fontSize: 14 }}>
            <div style={{ fontSize: 40, marginBottom: 8 }}>🖼️</div>
            Upload an image to preview and upscale
          </div>
        ) : (
          <div
            ref={containerRef}
            style={{
              position: 'relative',
              maxWidth: '100%',
              maxHeight: '100%',
              overflow: zoom !== 'fit' ? 'auto' : 'hidden',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            {/* Grid Cell Container holding both aligned canvases */}
            <div
              style={{
                display: 'grid',
                gridArea: '1 / 1',
                position: 'relative',
                maxWidth: '100%',
                maxHeight: '100%',
                transform: getZoomScale(),
                transformOrigin: 'center center',
                transition: 'transform 0.15s ease',
              }}
            >
              {/* Bottom Canvas: Processed HD Image */}
              {processedImage && (
                <canvas
                  ref={procCanvasRef}
                  style={{
                    gridArea: '1 / 1',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    display: 'block',
                  }}
                />
              )}

              {/* Top Canvas: Original Image (GPU Clipped dynamically on drag) */}
              <canvas
                ref={origCanvasRef}
                style={{
                  gridArea: '1 / 1',
                  maxWidth: '100%',
                  maxHeight: '100%',
                  display: 'block',
                  clipPath: processedImage ? `polygon(0 0, ${splitPos}% 0, ${splitPos}% 100%, 0 100%)` : 'none',
                }}
              />

              {/* Split Slider Divider Line & Handle */}
              {originalImage && processedImage && (
                <>
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      bottom: 0,
                      left: `${splitPos}%`,
                      width: 3,
                      background: '#ffffff',
                      boxShadow: '0 0 10px rgba(0,0,0,0.5)',
                      pointerEvents: 'none',
                      zIndex: 10,
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        background: '#ffffff',
                        color: '#0f172a',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.3)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 14,
                        fontWeight: 'bold',
                        cursor: 'ew-resize',
                      }}
                    >
                      ↔
                    </div>
                  </div>

                  <div
                    style={{
                      position: 'absolute',
                      top: 12,
                      left: 12,
                      background: 'rgba(15, 23, 42, 0.85)',
                      color: '#ffffff',
                      padding: '4px 12px',
                      borderRadius: 20,
                      fontSize: 11,
                      fontWeight: 600,
                      backdropFilter: 'blur(4px)',
                      zIndex: 10,
                    }}
                  >
                    ◀ Original ({Math.round(splitPos)}%) | Upscaled HD ▶
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
