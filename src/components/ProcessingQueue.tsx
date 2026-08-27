interface QueueProps {
  status: string
  progress?: number
  progressMessage?: string
}

export default function ProcessingQueue({ status, progress = 0, progressMessage = '' }: QueueProps) {
  const getStatusBadge = () => {
    switch (status) {
      case 'processing':
        return <span style={{ color: '#4f46e5', fontWeight: 600 }}>⚡ Upscaling in progress...</span>
      case 'done':
        return <span style={{ color: '#16a34a', fontWeight: 600 }}>✅ Upscale Complete</span>
      case 'error':
        return <span style={{ color: '#dc2626', fontWeight: 600 }}>❌ Inference Error</span>
      default:
        return <span style={{ color: '#64748b' }}>Ready</span>
    }
  }

  return (
    <div className="card">
      <h3>🚀 Status & Processing Queue</h3>
      <div style={{ fontSize: 13, marginBottom: 8 }}>
        Status: {getStatusBadge()}
      </div>

      {status === 'processing' && (
        <div style={{ marginTop: 14 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 500, color: '#475569', marginBottom: 6 }}>
            <span>{progressMessage || 'Processing neural tiles...'}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div style={{ width: '100%', height: 8, background: '#f1f5f9', borderRadius: 10, overflow: 'hidden' }}>
            <div
              style={{
                width: `${Math.min(100, Math.max(0, progress))}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #4f46e5, #06b6d4)',
                borderRadius: 10,
                transition: 'width 0.2s ease',
              }}
            />
          </div>
        </div>
      )}

      {status === 'done' && (
        <div style={{ fontSize: 12, color: '#15803d', marginTop: 10, background: '#f0fdf4', padding: '8px 12px', borderRadius: 8, border: '1px solid #bbf7d0' }}>
          🎉 Image upscaled with sharp detail! Use the zoom & slider in the Studio Canvas to inspect.
        </div>
      )}

      {status === 'error' && (
        <div style={{ fontSize: 12, color: '#b91c1c', marginTop: 10, background: '#fef2f2', padding: '8px 12px', borderRadius: 8, border: '1px solid #fecaca' }}>
          ⚠️ {progressMessage || 'Error occurred during image processing'}
        </div>
      )}
    </div>
  )
}
