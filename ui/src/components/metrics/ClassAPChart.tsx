import { useState, useEffect, useRef } from 'react'
import { VOC_CLASSES } from './mockData'

interface Props {
  data: Record<string, number>
}

const BAR_H = 7
const COL_ABOVE = '#8a9a6a'
const COL_BELOW = '#5a6a7a'

export const ClassAPChart = ({ data }: Props) => {
  const [ready, setReady] = useState(false)
  const [hovered, setHovered] = useState<string | null>(null)
  const trackRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 80)
    return () => clearTimeout(t)
  }, [])

  const sorted = [...VOC_CLASSES]
    .map(cls => ({ cls, ap: data[cls] ?? 0 }))
    .sort((a, b) => b.ap - a.ap)

  const mAP = sorted.reduce((s, x) => s + x.ap, 0) / sorted.length
  const mAPpct = `${(mAP * 100).toFixed(1)}%`

  return (
    <div
      data-testid="class-ap-chart"
      style={{
        padding: '18px 20px',
        display: 'flex', flexDirection: 'column', gap: 12,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.6px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)' }}>
          Per-Class AP
        </span>
        <span data-testid="map-summary" style={{ fontSize: '11px', color: COL_ABOVE, fontWeight: 700, fontFamily: 'monospace' }}>
          mAP {mAPpct}
        </span>
      </div>

      <div ref={trackRef} style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 4 }}>
        <svg
          aria-hidden="true"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible', zIndex: 1 }}
        >
          <line
            x1={`${mAP * 100}%`} x2={`${mAP * 100}%`}
            y1="0" y2="100%"
            stroke="rgba(255,255,255,0.18)" strokeWidth="1" strokeDasharray="3 3"
          />
          <circle cx={`${mAP * 100}%`} cy="0" r="3.5" fill="#fff" opacity="0.5" />
        </svg>

        {sorted.map(({ cls, ap }) => {
          const above = ap >= mAP
          const isHov = hovered === cls
          return (
            <div
              key={cls}
              data-testid={`class-row-${cls}`}
              style={{ display: 'flex', alignItems: 'center', gap: 8, position: 'relative', zIndex: 2 }}
              onMouseEnter={() => setHovered(cls)}
              onMouseLeave={() => setHovered(null)}
            >
              <span style={{
                width: 76, textAlign: 'right', fontSize: '9.5px', flexShrink: 0,
                color: isHov ? 'var(--text-primary)' : 'rgba(255,255,255,0.3)',
                transition: 'color 0.15s',
              }}>
                {cls}
              </span>

              <div style={{ flex: 1, height: BAR_H, background: 'rgba(255,255,255,0.04)', position: 'relative', overflow: 'hidden' }}>
                <div
                  data-testid={`bar-${cls}`}
                  style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: ready ? `${ap * 100}%` : '0%',
                    background: above ? COL_ABOVE : COL_BELOW,
                    opacity: isHov ? 0.85 : 0.5,
                    transition: 'width 0.6s cubic-bezier(0.22,1,0.36,1), opacity 0.15s',
                  }}
                />
              </div>

              <span
                data-testid={`score-${cls}`}
                style={{
                  width: 34, fontSize: '9px', fontFamily: 'monospace', flexShrink: 0,
                  textAlign: 'right',
                  color: isHov ? '#e8eae9' : above ? 'rgba(138,154,106,0.7)' : 'rgba(255,255,255,0.2)',
                  transition: 'color 0.15s',
                }}
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
