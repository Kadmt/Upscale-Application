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

  const profiles = [
    {
      id: 'document8k' as const,
      title: '📜 8K Văn Bản & Document',
      desc: 'Scan PDF, Sách, Giấy khen, Chữ viết tay',
    },
    {
      id: 'general' as const,
      title: '🖼️ Ảnh Phong Cảnh & Vật Thể',
      desc: 'Phong cảnh, Vật thể, Đồ họa 2D, Kiến trúc',
    },
    {
      id: 'portrait' as const,
      title: '👤 Chân Dung & Khuôn Mặt AI',
      desc: 'Ảnh người, Ánh mắt, Gọng kính, Nước da',
    },
  ]

  return (
    <div className="card" style={{ padding: 18 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h3 style={{ margin: 0, fontSize: 13, textTransform: 'uppercase', letterSpacing: 0.5, color: '#0f172a' }}>
          ⚡ THIẾT LẬP UPSCALE (STUDIO SETTINGS)
        </h3>
      </div>

      {/* Profile Selection */}
      <div className="control-group">
        <label className="control-label" style={{ fontWeight: 600, fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: 8, display: 'block' }}>
          1. CHỌN CHẾ ĐỘ XỬ LÝ (PROFILE)
        </label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {profiles.map((p) => {
            const isActive = mode === p.id
            return (
              <button
                key={p.id}
                type="button"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  padding: '10px 12px',
                  borderRadius: 6,
                  border: isActive ? '2px solid #0f172a' : '1px solid #e2e8f0',
                  background: isActive ? '#f8fafc' : '#ffffff',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                }}
                onClick={() => onModeChange(p.id)}
              >
                <div style={{ fontSize: 13, fontWeight: 700, color: isActive ? '#0f172a' : '#334155', marginBottom: 2 }}>
                  {p.title}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.3 }}>
                  {p.desc}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Target Scale Selection */}
      <div className="control-group" style={{ marginTop: 16 }}>
        <label className="control-label" style={{ fontWeight: 600, fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: 8, display: 'block' }}>
          2. TỶ LỆ PHÓNG TO (TARGET SCALE)
        </label>
        <div className="pill-group" style={{ background: '#f8fafc', border: '1px solid #e2e8f0', padding: 3, borderRadius: 6 }}>
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

      {/* Fine-Tuning Controls */}
      <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid #e2e8f0' }}>
        <button
          type="button"
          className="btn-toggle-advanced"
          style={{ background: '#f8fafc', border: '1px solid #e2e8f0', padding: '8px 12px', borderRadius: 6, width: '100%', fontSize: 12, fontWeight: 600, color: '#334155', cursor: 'pointer', textAlign: 'left', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <span>⚙️ Tinh chỉnh nâng cao (Fine-Tuning)</span>
          <span style={{ fontSize: 10 }}>{showAdvanced ? '▲ Thu gọn' : '▼ Mở rộng'}</span>
        </button>

        {showAdvanced && (
          <div className="advanced-panel" style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6, padding: 12, marginTop: 8 }}>
            {mode === 'document8k' && (
              <div className="control-group" style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <label className="control-label" style={{ margin: 0, fontSize: 11, fontWeight: 600 }}>✒️ Độ Đậm Nét Chữ (Ink Factor)</label>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#0f172a' }}>{darkness}%</span>
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

            <div className="control-group" style={{ margin: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <label className="control-label" style={{ margin: 0, fontSize: 11, fontWeight: 600 }}>✨ Tăng Cường Độ Nét (Sharpness)</label>
                <span style={{ fontSize: 11, fontWeight: 700, color: '#0f172a' }}>{sharpness}%</span>
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
