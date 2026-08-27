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
        <label className="control-label" style={{ fontWeight: 600, fontSize: 12, marginBottom: 8, display: 'block', color: '#475569' }}>
          CHỌN CHẾ ĐỘ XỬ LÝ (ENHANCEMENT PROFILE)
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8 }}>
          <button
            type="button"
            className={`profile-card ${mode === 'document8k' ? 'active' : ''}`}
            style={{
              padding: '10px 10px',
              borderRadius: 6,
              border: mode === 'document8k' ? '2px solid #0f172a' : '1px solid #e2e8f0',
              background: mode === 'document8k' ? '#f8fafc' : '#ffffff',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onClick={() => onModeChange('document8k')}
          >
            <div style={{ fontSize: 13, fontWeight: 700, color: mode === 'document8k' ? '#0f172a' : '#334155', marginBottom: 2 }}>📜 8K Văn Bản</div>
            <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.3 }}>Scan PDF, Sách, Chữ viết tay</div>
          </button>

          <button
            type="button"
            className={`profile-card ${mode === 'general' ? 'active' : ''}`}
            style={{
              padding: '10px 10px',
              borderRadius: 6,
              border: mode === 'general' ? '2px solid #0f172a' : '1px solid #e2e8f0',
              background: mode === 'general' ? '#f8fafc' : '#ffffff',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onClick={() => onModeChange('general')}
          >
            <div style={{ fontSize: 13, fontWeight: 700, color: mode === 'general' ? '#0f172a' : '#334155', marginBottom: 2 }}>🖼️ Ảnh & Cảnh</div>
            <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.3 }}>Phong cảnh, Vật thể, Kiến trúc</div>
          </button>

          <button
            type="button"
            className={`profile-card ${mode === 'portrait' ? 'active' : ''}`}
            style={{
              padding: '10px 10px',
              borderRadius: 6,
              border: mode === 'portrait' ? '2px solid #0f172a' : '1px solid #e2e8f0',
              background: mode === 'portrait' ? '#f8fafc' : '#ffffff',
              textAlign: 'left',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onClick={() => onModeChange('portrait')}
          >
            <div style={{ fontSize: 13, fontWeight: 700, color: mode === 'portrait' ? '#0f172a' : '#334155', marginBottom: 2 }}>👤 Chân Dung AI</div>
            <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.3 }}>Khuôn mặt, Ánh mắt, Làn da</div>
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
