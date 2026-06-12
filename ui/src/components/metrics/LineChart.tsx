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

const VW = 640, VH = 240, LP = { top: 24, right: 18, bottom: 38, left: 52 }
const COLS = ['#00d4a0', '#e88548', '#7c9ef5']

export const LineChart = ({ title, sub, series }: Props) => {
  const [hover, setHover] = useState<HoverState | null>(null)

  const allVals = series.flatMap(s => s.data)
  const xSc = d3.scaleLinear().domain([1, 200]).range([LP.left, VW - LP.right])
  const pad = (Math.max(...allVals) - Math.min(...allVals)) * 0.07
  const ySc = d3.scaleLinear()
    .domain([Math.min(...allVals) - pad, Math.max(...allVals) + pad])
    .range([VH - LP.bottom, LP.top])

  const mkLine = d3.line<number>().x((_, i) => xSc(i + 1)).y(d => ySc(d)).curve(d3.curveCatmullRom.alpha(0.5))
  const mkArea = d3.area<number>().x((_, i) => xSc(i + 1)).y0(VH - LP.bottom).y1(d => ySc(d)).curve(d3.curveCatmullRom.alpha(0.5))
  const yTicks = ySc.ticks(5)
  const xTicks = [1, 40, 80, 120, 160, 200]

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const svgX = (e.clientX - rect.left) / rect.width * VW
    if (svgX < LP.left || svgX > VW - LP.right) { setHover(null); return }
    const epoch = Math.max(1, Math.min(200, Math.round(xSc.invert(svgX))))
    setHover({
      epoch,
      cx: xSc(epoch),
      pct: xSc(epoch) / VW,
      vals: series.map((s, i) => ({ label: s.label, v: s.data[epoch - 1], col: COLS[i] })),
    })
  }

  return (
    <div
      data-testid="line-chart"
      style={{
        background: '#0f1714', border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 16, padding: '20px 22px 14px',
        display: 'flex', flexDirection: 'column', gap: 14,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span data-testid="chart-title" style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>{title}</span>
          {sub && <span data-testid="chart-sub" style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{sub}</span>}
        </div>
        <div style={{ display: 'flex', gap: 14 }}>
          {series.map((s, i) => (
            <div key={i} data-testid={`legend-${s.label}`} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 16, height: 2, background: COLS[i], borderRadius: 2 }} />
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
          <defs>
            {series.map((_, i) => (
              <linearGradient key={i} id={`lcg${i}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={COLS[i]} stopOpacity="0.30" />
                <stop offset="80%" stopColor={COLS[i]} stopOpacity="0.02" />
              </linearGradient>
            ))}
            <clipPath id="cc"><rect x={LP.left} y={LP.top} width={VW - LP.left - LP.right} height={VH - LP.top - LP.bottom} /></clipPath>
          </defs>

          {yTicks.map((t, i) => (
            <line key={i} x1={LP.left} x2={VW - LP.right} y1={ySc(t)} y2={ySc(t)}
              stroke="rgba(255,255,255,0.045)" strokeWidth="1" strokeDasharray="4 3" />
          ))}

          {series.map((s, i) => (
            <path key={i} d={mkArea(s.data) ?? ''} fill={`url(#lcg${i})`} clipPath="url(#cc)" />
          ))}
          {series.map((s, i) => (
            <path key={i} d={mkLine(s.data) ?? ''} fill="none"
              stroke={COLS[i]} strokeWidth="1.8" clipPath="url(#cc)" strokeLinejoin="round" />
          ))}

          <line x1={LP.left} x2={VW - LP.right} y1={VH - LP.bottom} y2={VH - LP.bottom}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

          {xTicks.map(t => (
            <text key={t} x={xSc(t)} y={VH - LP.bottom + 14} textAnchor="middle"
              fontSize="9" fill="var(--text-tertiary)">{t}</text>
          ))}

          {hover && (
            <line data-testid="crosshair"
              x1={hover.cx} x2={hover.cx} y1={LP.top} y2={VH - LP.bottom}
              stroke="rgba(255,255,255,0.18)" strokeWidth="1" strokeDasharray="3 2" />
          )}
        </svg>

        {hover && (
          <div data-testid="hover-popover" style={{
            position: 'absolute',
            left: `calc(${hover.pct * 100}% + ${hover.pct > 0.68 ? -144 : 14}px)`,
            top: '18px', pointerEvents: 'none',
            background: 'rgba(8,14,12,0.96)',
            border: '1px solid rgba(255,255,255,0.13)',
            borderRadius: 10, padding: '9px 13px', minWidth: 128, zIndex: 20,
          }}>
            <div style={{ fontSize: '9px', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-tertiary)', marginBottom: 7 }}>
              Epoch {hover.epoch}
            </div>
            {hover.vals.map(v => (
              <div key={v.label} style={{ display: 'flex', justifyContent: 'space-between', gap: 18, marginBottom: 4 }}>
                <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{v.label}</span>
                <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{v.v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
