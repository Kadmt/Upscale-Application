import React, { useState } from 'react'

interface SettingsProps {
  scale: number
  sharpness: number
  darkness: number
  mode: 'general' | 'document8k' | 'portrait'
  onScaleChange: (scale: number) => void
  onSharpnessChange: (sharpness: number) => void
  onDarknessChange: (darkness: number) => void
  onModeChange: (mode: 'general' | 'document8k' | 'portrait') => void
}

export default function Settings({
  scale,
  sharpness,
  darkness,
  mode,
  onScaleChange,
  onSharpnessChange,
  onDarknessChange,
  onModeChange,
}: SettingsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  return (
    <div className="settings-panel">
      <div className="panel-header">
        <span className="panel-title">⚡ Upscale Studio Settings</span>
        <button
          type="button"
          className="btn-secondary"
          style={{ fontSize: 12, padding: '4px 10px' }}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '⚙️ Hide Advanced' : '⚙️ Advanced Controls'}
        </button>
      </div>

      {/* Mode Selection Profile */}
      <div className="control-group">
        <label className="control-label">Chế độ xử lý (Enhancement Profile)</label>
        <div className="pill-group">
          <button
            type="button"
            className={`pill-btn ${mode === 'document8k' ? 'active' : ''}`}
            onClick={() => onModeChange('document8k')}
          >
            📜 8K Document Scan
          </button>
          <button
            type="button"
            className={`pill-btn ${mode === 'general' ? 'active' : ''}`}
            onClick={() => onModeChange('general')}
          >
            🖼️ Natural Photo
          </button>
          <button
            type="button"
            className={`pill-btn ${mode === 'portrait' ? 'active' : ''}`}
            onClick={() => onModeChange('portrait')}
          >
            👤 AI Portrait & Face
          </button>
        </div>
      </div>

      {/* Scale Selection */}
      <div className="control-group">
        <label className="control-label">Tỷ lệ phóng to (Target Scale)</label>
        <div className="pill-group">
          <button
            type="button"
            className={`pill-btn ${scale === 2 ? 'active' : ''}`}
            onClick={() => onScaleChange(2)}
          >
            2x HD
          </button>
          <button
            type="button"
            className={`pill-btn ${scale === 4 ? 'active' : ''}`}
            onClick={() => onScaleChange(4)}
          >
            4x Ultra HD
          </button>
          <button
            type="button"
            className={`pill-btn ${scale === 8 ? 'active' : ''}`}
            onClick={() => onScaleChange(8)}
          >
            8x Extreme 8K
          </button>
        </div>
      </div>

      {/* Collapsible Advanced Fine-Tuning */}
      <div style={{ marginTop: 12 }}>
        <button
          type="button"
          className="btn-toggle-advanced"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced
            ? '▼ Nâng cao (Thu gọn)'
            : mode === 'document8k'
            ? '▶ Tùy chỉnh nâng cao (Độ đậm / Độ nét)'
            : '▶ Tùy chỉnh nâng cao (Độ nét chi tiết)'}
        </button>

        {showAdvanced && (
          <div className="advanced-panel">
            {/* Show Text Darkness control ONLY for Document mode */}
            {mode === 'document8k' && (
              <div className="control-group" style={{ marginTop: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <label className="control-label" style={{ margin: 0 }}>✒️ Độ Đậm Nét Chữ</label>
                  <span style={{ fontSize: 12, fontWeight: 700, color: '#059669' }}>{darkness}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="40"
                  step="5"
                  value={darkness}
                  onChange={(e) => onDarknessChange(Number(e.target.value))}
                />
              </div>
            )}

            <div className="control-group" style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <label className="control-label" style={{ margin: 0 }}>✨ Sharpness Edge Filter</label>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#4f46e5' }}>{sharpness}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={sharpness}
                onChange={(e) => onSharpnessChange(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
