import { useState } from 'react'
import * as d3 from 'd3'

interface Series {
  label: string
  data: number[]
}

interface HoverState {
  epoch: number
  cx: number
  pct: number
  vals: { label: string; v: number; col: string }[]
}

interface Props {
  title: string
  sub?: string
  series: Series[]
}

const VW = 640, VH = 200, LP = { top: 32, right: 18, bottom: 42, left: 48 }
const COLS = ['#8a9a6a', '#6a7a8a', '#7c9ef5']

export const LineChart = ({ title, sub, series }: Props) => {
  const [hover, setHover] = useState<HoverState | null>(null)

  const n = series[0]?.data.length ?? 200
  const allVals = series.flatMap(s => s.data)
  const xSc = d3.scaleLinear().domain([0, n - 1]).range([LP.left, VW - LP.right])
  const yMax = Math.max(...allVals)
  const ySc = d3.scaleLinear().domain([0, yMax * 1.08]).range([VH - LP.bottom, LP.top])

  const lineGen = d3.line<number>().x((_, i) => xSc(i)).y(v => ySc(v))
  const areaGen = d3.area<number>().x((_, i) => xSc(i)).y0(VH - LP.bottom).y1(v => ySc(v))

  const yTicks = ySc.ticks(4)
  const markerEpochs = (() => {
    const peaks: number[] = []
    series.forEach(s => {
      const maxIdx = s.data.indexOf(Math.max(...s.data))
      const minIdx = s.data.indexOf(Math.min(...s.data))
      peaks.push(maxIdx, minIdx)
    })
    return [...new Set(peaks)].sort((a, b) => a - b).slice(0, 4)
  })()

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const svgX = (e.clientX - rect.left) / rect.width * VW
    if (svgX < LP.left || svgX > VW - LP.right) { setHover(null); return }
    const idx = Math.max(0, Math.min(n - 1, Math.round(xSc.invert(svgX))))
    setHover({
      epoch: idx + 1,
      cx: xSc(idx),
      pct: xSc(idx) / VW,
      vals: series.map((s, i) => ({ label: s.label, v: s.data[idx], col: COLS[i] })),
    })
  }

  return (
    <div
      data-testid="line-chart"
      style={{ display: 'flex', flexDirection: 'column', gap: 10 }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', padding: '0 4px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span data-testid="chart-title" style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.2px' }}>{title}</span>
          {sub && <span data-testid="chart-sub" style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{sub}</span>}
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          {series.map((s, i) => (
            <div key={i} data-testid={`legend-${s.label}`} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: COLS[i], opacity: 0.8 }} />
              <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative' }}>
        <svg
          data-testid="chart-svg"
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
              fontSize="8.5" fill="rgba(255,255,255,0.25)">{t.toFixed(2)}</text>
          ))}

          {series.map((s, si) => (
            <g key={si}>
              <path d={areaGen(s.data) ?? ''} fill={COLS[si]} opacity={0.08} stroke="none" />
              <path d={lineGen(s.data) ?? ''} fill="none" stroke={COLS[si]} strokeWidth={1.5} />
            </g>
          ))}

          {hover && series.map((s, si) => (
            <circle key={si} cx={hover.cx} cy={ySc(s.data[hover.epoch - 1])} r={3}
              fill={COLS[si]} opacity={0.9} />
          ))}

          <line x1={LP.left} x2={VW - LP.right} y1={VH - LP.bottom} y2={VH - LP.bottom}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

          {markerEpochs.map(idx => {
            const cx = xSc(idx)
            const label = String(idx + 1)
            return (
              <g key={idx}>
                <line x1={cx} x2={cx} y1={LP.top} y2={VH - LP.bottom}
                  stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="3 3" />
                <circle cx={cx} cy={VH - LP.bottom} r={3.5} fill="#fff" opacity={0.7} />
                <text x={cx} y={VH - LP.bottom + 14} textAnchor="middle"
                  fontSize="8.5" fill="rgba(255,255,255,0.4)">{label}</text>
              </g>
            )
          })}

          {hover && (
            <g>
              <line data-testid="crosshair"
                x1={hover.cx} x2={hover.cx} y1={LP.top} y2={VH - LP.bottom}
                stroke="rgba(255,255,255,0.3)" strokeWidth="1" strokeDasharray="3 2" />
              <circle cx={hover.cx} cy={VH - LP.bottom} r={4} fill="#fff" opacity={0.9} />
              <text x={hover.cx} y={VH - LP.bottom + 14} textAnchor="middle"
                fontSize="8.5" fill="rgba(255,255,255,0.6)">{hover.epoch}</text>
            </g>
          )}
        </svg>

        {hover && (
          <div data-testid="hover-popover" style={{
            position: 'absolute',
            left: `calc(${hover.pct * 100}% + ${hover.pct > 0.68 ? -144 : 14}px)`,
            top: '18px', pointerEvents: 'none',
            backdropFilter: 'blur(20px) saturate(1.6)',
            WebkitBackdropFilter: 'blur(20px) saturate(1.6)',
            background: 'rgba(8,16,12,0.82)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 8, padding: '8px 12px', minWidth: 120, zIndex: 20,
          }}>
            <div style={{ fontSize: '9px', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-tertiary)', marginBottom: 6 }}>
              Epoch {hover.epoch}
            </div>
            {hover.vals.map(v => (
              <div key={v.label} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 3 }}>
                <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{v.label}</span>
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#fff', fontFamily: 'monospace' }}>{v.v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
