import { useState } from 'react'
import * as d3 from 'd3'

interface HoverState {
  epoch: number
  cx: number
  pct: number
  score: number
  avgDet: number
  zeroDet: number
}

interface Props {
  meanScore: number[]
  avgDetections: number[]
  zeroDetRatio: number[]
}

const VW = 640, VH = 180
const LP = { top: 28, right: 18, bottom: 38, left: 44 }

const COLS = {
  score:   '#8a9a6a',
  avgDet:  '#6a7a8a',
  zeroDet: '#8a6a6a',
}

export const NMSHealthChart = ({ meanScore, avgDetections, zeroDetRatio }: Props) => {
  const [hover, setHover] = useState<HoverState | null>(null)

  const n = meanScore.length

  const xSc = d3.scaleLinear().domain([0, n - 1]).range([LP.left, VW - LP.right])

  const yScLeft = d3.scaleLinear().domain([0, 1]).range([VH - LP.bottom, LP.top])

  const detMax = Math.max(...avgDetections) * 1.15
  const yScRight = d3.scaleLinear().domain([0, detMax]).range([VH - LP.bottom, LP.top])

  const line = (data: number[], ySc: d3.ScaleLinear<number, number>) =>
    data.map((v, i) => `${xSc(i)},${ySc(v)}`).join(' ')

  const yTicksLeft = yScLeft.ticks(3)
  const yTicksRight = yScRight.ticks(3)

  const BAR_W = Math.max(1, (xSc(1) - xSc(0)) * 0.55)

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const svgX = (e.clientX - rect.left) / rect.width * VW
    if (svgX < LP.left || svgX > VW - LP.right) { setHover(null); return }
    const idx = Math.max(0, Math.min(n - 1, Math.round(xSc.invert(svgX))))
    setHover({
      epoch: idx + 1,
      cx: xSc(idx),
      pct: xSc(idx) / VW,
      score: meanScore[idx],
      avgDet: avgDetections[idx],
      zeroDet: zeroDetRatio[idx],
    })
  }

  return (
    <div data-testid="nms-health-chart" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', padding: '0 4px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.2px' }}>
            Detection Health
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>
            NMS output quality per eval epoch
          </span>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          {([['score', 'mean score'], ['avgDet', 'avg det'], ['zeroDet', 'zero det %']] as const).map(([key, label]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: COLS[key], opacity: 0.8 }} />
              <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative' }}>
        <svg
          data-testid="nms-svg"
          viewBox={`0 0 ${VW} ${VH}`}
          width="100%"
          style={{ display: 'block', cursor: 'crosshair', overflow: 'visible' }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {yTicksLeft.map((t, i) => (
            <line key={i} x1={LP.left} x2={VW - LP.right} y1={yScLeft(t)} y2={yScLeft(t)}
              stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
          ))}

          {yTicksLeft.map((t, i) => (
            <text key={i} x={LP.left - 5} y={yScLeft(t) + 3.5} textAnchor="end"
              fontSize="8.5" fill="rgba(255,255,255,0.22)">{t.toFixed(1)}</text>
          ))}

          {yTicksRight.map((t, i) => (
            <text key={i} x={VW - LP.right + 5} y={yScRight(t) + 3.5} textAnchor="start"
              fontSize="8.5" fill={`${COLS.avgDet}99`}>{t.toFixed(0)}</text>
          ))}

          {avgDetections.map((v, i) => {
            const x = xSc(i) - BAR_W / 2
            const y = yScRight(v)
            const h = (VH - LP.bottom) - y
            return h > 0 ? (
              <rect key={i} x={x} y={y} width={BAR_W} height={h}
                fill={COLS.avgDet}
                opacity={hover?.epoch === i + 1 ? 0.6 : 0.25}
                style={{ transition: 'opacity 0.1s' }}
              />
            ) : null
          })}

          <line x1={LP.left} x2={VW - LP.right} y1={VH - LP.bottom} y2={VH - LP.bottom}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1" />

          <polyline
            points={line(zeroDetRatio, yScLeft)}
            fill="none" stroke={COLS.zeroDet} strokeWidth="1.2" opacity="0.55" strokeDasharray="4 3"
          />

          <polyline
            points={line(meanScore, yScLeft)}
            fill="none" stroke={COLS.score} strokeWidth="1.5" opacity="0.8"
          />

          {(() => {
            const bestIdx = meanScore.indexOf(Math.max(...meanScore))
            return (
              <g>
                <line x1={xSc(bestIdx)} x2={xSc(bestIdx)} y1={LP.top} y2={VH - LP.bottom}
                  stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3 3" />
                <circle cx={xSc(bestIdx)} cy={yScLeft(meanScore[bestIdx])} r="3.5" fill="#fff" opacity={0.6} />
              </g>
            )
          })()}

          {hover && (
            <g>
              <line x1={hover.cx} x2={hover.cx} y1={LP.top} y2={VH - LP.bottom}
                stroke="rgba(255,255,255,0.25)" strokeWidth="1" strokeDasharray="3 2" />
              <circle cx={hover.cx} cy={yScLeft(hover.score)} r="3.5" fill={COLS.score} opacity={0.9} />
              <circle cx={hover.cx} cy={yScLeft(hover.zeroDet)} r="2.5" fill={COLS.zeroDet} opacity={0.8} />
              <circle cx={hover.cx} cy={yScRight(hover.avgDet)} r="2.5" fill={COLS.avgDet} opacity={0.8} />
            </g>
          )}
        </svg>

        {hover && (
          <div data-testid="nms-popover" style={{
            position: 'absolute',
            left: `calc(${hover.pct * 100}% + ${hover.pct > 0.68 ? -160 : 14}px)`,
            top: '16px', pointerEvents: 'none',
            backdropFilter: 'blur(20px) saturate(1.6)',
            WebkitBackdropFilter: 'blur(20px) saturate(1.6)',
            background: 'rgba(8,16,12,0.82)',
            border: '1px solid rgba(255,255,255,0.09)',
            borderRadius: 8, padding: '8px 12px', minWidth: 140, zIndex: 20,
          }}>
            <div style={{ fontSize: '9px', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-tertiary)', marginBottom: 6 }}>
              Epoch {hover.epoch}
            </div>
            {([
              ['mean score',  hover.score.toFixed(3),    COLS.score],
              ['avg det',     hover.avgDet.toFixed(1),   COLS.avgDet],
              ['zero det %',  (hover.zeroDet * 100).toFixed(1) + '%', COLS.zeroDet],
            ] as const).map(([label, val, col]) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 3 }}>
                <span style={{ fontSize: '10px', color: col as string, opacity: 0.8 }}>{label}</span>
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#fff', fontFamily: 'monospace' }}>{val}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
