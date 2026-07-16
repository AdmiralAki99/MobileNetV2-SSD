import { useState } from 'react'
import * as d3 from 'd3'

interface HoverState {
  epoch: number
  cx: number
  pct: number
  lr: number
}

interface Props {
  data: number[]
}

const VW = 640, VH = 160
const LP = { top: 28, right: 18, bottom: 38, left: 58 }
const COL = '#8a9a6a'

export const LRChart = ({ data }: Props) => {
  const [hover, setHover] = useState<HoverState | null>(null)

  const n = data.length
  const xSc = d3.scaleLinear().domain([0, n - 1]).range([LP.left, VW - LP.right])
  const yMax = Math.max(...data)
  const yMin = Math.min(...data)
  const ySc = d3.scaleLinear().domain([0, yMax * 1.1]).range([VH - LP.bottom, LP.top])

  const points = data.map((v, i) => `${xSc(i)},${ySc(v)}`).join(' ')

  const yTicks = ySc.ticks(3)
  const fmtLR = (v: number) => {
    if (v === 0) return '0'
    if (v >= 0.001) return v.toFixed(4)
    return v.toExponential(1)
  }

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const svgX = (e.clientX - rect.left) / rect.width * VW
    if (svgX < LP.left || svgX > VW - LP.right) { setHover(null); return }
    const idx = Math.max(0, Math.min(n - 1, Math.round(xSc.invert(svgX))))
    setHover({ epoch: idx + 1, cx: xSc(idx), pct: xSc(idx) / VW, lr: data[idx] })
  }

  const warmupEnd = data.findIndex((v, i) => i > 0 && data[i] < data[i - 1])
  const markerEpochs = warmupEnd > 0 ? [warmupEnd] : []

  return (
    <div data-testid="lr-chart" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', padding: '0 4px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.2px' }}>
            Learning Rate
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>
            warmup → cosine decay
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: COL, opacity: 0.8 }} />
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>lr</span>
        </div>
      </div>

      <div style={{ position: 'relative' }}>
        <svg
          data-testid="lr-svg"
          viewBox={`0 0 ${VW} ${VH}`}
          width="100%"
          style={{ display: 'block', cursor: 'crosshair', overflow: 'visible' }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {yTicks.map((t, i) => (
            <line key={i} x1={LP.left} x2={VW - LP.right} y1={ySc(t)} y2={ySc(t)}
              stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
          ))}

          {yTicks.map((t, i) => (
            <text key={i} x={LP.left - 6} y={ySc(t) + 3.5} textAnchor="end"
              fontSize="8" fill="rgba(255,255,255,0.22)" fontFamily="monospace">
              {fmtLR(t)}
            </text>
          ))}

          <defs>
            <linearGradient id="lr-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={COL} stopOpacity="0.18" />
              <stop offset="100%" stopColor={COL} stopOpacity="0.01" />
            </linearGradient>
          </defs>
          <polygon
            points={`${LP.left},${VH - LP.bottom} ${points} ${xSc(n - 1)},${VH - LP.bottom}`}
            fill="url(#lr-fill)"
          />

          <polyline
            points={points}
            fill="none"
            stroke={COL}
            strokeWidth="1.5"
            opacity="0.75"
          />

          <line x1={LP.left} x2={VW - LP.right} y1={VH - LP.bottom} y2={VH - LP.bottom}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

          {markerEpochs.map(idx => {
            const cx = xSc(idx)
            return (
              <g key={idx}>
                <line x1={cx} x2={cx} y1={LP.top} y2={VH - LP.bottom}
                  stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="3 3" />
                <circle cx={cx} cy={ySc(data[idx])} r="3.5" fill="#fff" opacity={0.6} />
                <text x={cx + 5} y={LP.top + 10} fontSize="8" fill="rgba(255,255,255,0.3)" fontFamily="monospace">
                  peak
                </text>
              </g>
            )
          })}

          {(() => {
            const minIdx = data.indexOf(yMin)
            const cx = xSc(minIdx)
            return (
              <g>
                <circle cx={cx} cy={ySc(yMin)} r="3" fill="#fff" opacity={0.35} />
              </g>
            )
          })()}

          {hover && (
            <g>
              <line x1={hover.cx} x2={hover.cx} y1={LP.top} y2={VH - LP.bottom}
                stroke="rgba(255,255,255,0.25)" strokeWidth="1" strokeDasharray="3 2" />
              <circle cx={hover.cx} cy={ySc(hover.lr)} r="3.5" fill="#fff" opacity={0.85} />
            </g>
          )}
        </svg>

        {hover && (
          <div data-testid="lr-popover" style={{
            position: 'absolute',
            left: `calc(${hover.pct * 100}% + ${hover.pct > 0.72 ? -140 : 12}px)`,
            top: '16px', pointerEvents: 'none',
            backdropFilter: 'blur(20px) saturate(1.6)',
            WebkitBackdropFilter: 'blur(20px) saturate(1.6)',
            background: 'rgba(8,16,12,0.82)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 8, padding: '8px 12px', minWidth: 110, zIndex: 20,
          }}>
            <div style={{ fontSize: '9px', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-tertiary)', marginBottom: 5 }}>
              Epoch {hover.epoch}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>lr</span>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#fff', fontFamily: 'monospace' }}>
                {hover.lr.toExponential(3)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
