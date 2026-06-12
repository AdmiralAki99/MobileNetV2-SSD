import { useState, useEffect } from 'react'
import { VOC_CLASSES } from './mockData'

interface Props {
  data: Record<string, number>
}

export const ClassAPChart = ({ data }: Props) => {
  const [ready, setReady] = useState(false)
  const [hovered, setHovered] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 80)
    return () => clearTimeout(t)
  }, [])

  const sorted = [...VOC_CLASSES]
    .map(cls => ({ cls, ap: data[cls] ?? 0 }))
    .sort((a, b) => b.ap - a.ap)

  const mAP = sorted.reduce((s, x) => s + x.ap, 0) / sorted.length

  return (
    <div
      data-testid="class-ap-chart"
      style={{
        background: '#0f1714', border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 16, padding: '20px 22px',
        display: 'flex', flexDirection: 'column', gap: 14,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Per-Class AP</span>
        <span data-testid="map-summary" style={{ fontSize: '11px', color: 'var(--accent)', fontWeight: 600 }}>
          mAP {(mAP * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {sorted.map(({ cls, ap }) => {
          const above = ap >= mAP
          const isHovered = hovered === cls
          return (
            <div
              key={cls}
              data-testid={`class-row-${cls}`}
              style={{ display: 'flex', alignItems: 'center', gap: 10 }}
              onMouseEnter={() => setHovered(cls)}
              onMouseLeave={() => setHovered(null)}
            >
              <span style={{ width: 78, textAlign: 'right', fontSize: '10px', flexShrink: 0, color: isHovered ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {cls}
              </span>
              <div style={{ flex: 1, height: 9, background: 'rgba(255,255,255,0.05)', borderRadius: 6, position: 'relative', overflow: 'hidden' }}>
                <div
                  data-testid={`bar-${cls}`}
                  style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: ready ? `${ap * 100}%` : '0%',
                    background: above ? 'rgba(0,212,160,0.7)' : 'rgba(232,133,72,0.65)',
                    borderRadius: 6,
                    transition: 'width 0.7s cubic-bezier(0.22,1,0.36,1)',
                  }}
                />
              </div>
              <span
                data-testid={`score-${cls}`}
                style={{ width: 34, fontSize: '9.5px', fontFamily: 'monospace', flexShrink: 0, textAlign: 'right', color: above ? 'var(--accent)' : 'var(--text-tertiary)' }}
              >
                {(ap * 100).toFixed(1)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
