import { useEffect, useRef } from 'react'

interface Props {
  value: number
  label: string
  sublabel?: string
  color?: string
  size?: number
  testId?: string
}

const START_DEG = 225
const SWEEP_DEG = 270
const TO_RAD    = Math.PI / 180

function polarXY(cx: number, cy: number, r: number, deg: number) {
  const rad = deg * TO_RAD
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)] as const
}

function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  const delta = endDeg - startDeg
  if (delta <= 0.01) return ''
  const [x1, y1] = polarXY(cx, cy, r, startDeg)
  const [x2, y2] = polarXY(cx, cy, r, endDeg)
  const large = delta > 180 ? 1 : 0
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`
}

export const ArcGauge = ({ value, label, sublabel, color = '#00d4a0', size = 68, testId }: Props) => {
  const pct      = Math.max(0, Math.min(1, value))
  const cx       = size / 2
  const cy       = size / 2
  const r        = size * 0.36

  const trackEnd = START_DEG + SWEEP_DEG
  const valueEnd = START_DEG + SWEEP_DEG * pct

  const valueArcLen = r * SWEEP_DEG * TO_RAD * pct

  const [ax, ay] = polarXY(cx, cy, r,     valueEnd)
  const [bx, by] = polarXY(cx, cy, r + 8, valueEnd)

  const arcRef = useRef<SVGPathElement>(null)
  const tipRef = useRef<SVGGElement>(null)

  useEffect(() => {
    const arc = arcRef.current
    const tip = tipRef.current
    if (!arc) return

    if (pct === 0) return

    arc.style.transition       = 'none'
    arc.style.strokeDasharray  = `${valueArcLen}`
    arc.style.strokeDashoffset = `${valueArcLen}`
    if (tip) { tip.style.transition = 'none'; tip.style.opacity = '0' }

    requestAnimationFrame(() => requestAnimationFrame(() => {
      arc.style.transition       = 'stroke-dashoffset 1.1s cubic-bezier(0.25,0.46,0.45,0.94)'
      arc.style.strokeDashoffset = '0'
      if (tip) {
        tip.style.transition = 'opacity 0.25s ease 1.05s'
        tip.style.opacity    = '1'
      }
    }))
  }, [pct, valueArcLen])

  const filterId = `ag-glow-${label.replace(/\W/g, '')}`

  const displayVal = pct > 0 ? Math.round(pct * 100).toString() : '—'

  return (
    <div
      data-testid={testId}
      style={{ display: 'flex', alignItems: 'center', gap: 12 }}
    >
      <svg
        width={size} height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ flexShrink: 0, overflow: 'visible' }}
      >
        <defs>
          <filter id={filterId} x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="2" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <circle cx={cx} cy={cy} r={r + 5} fill="rgba(0,0,0,0.28)" />

        <path
          d={describeArc(cx, cy, r, START_DEG, trackEnd)}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />

        {pct > 0 && (
          <path
            ref={arcRef}
            d={describeArc(cx, cy, r, START_DEG, valueEnd)}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
            filter={`url(#${filterId})`}
            style={{ willChange: 'stroke-dashoffset' }}
          />
        )}

        {pct > 0 && (
          <g ref={tipRef} style={{ opacity: 0 }}>
            <line
              x1={ax} y1={ay} x2={bx} y2={by}
              stroke={color}
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <circle
              cx={bx} cy={by} r={2}
              fill={color}
              filter={`url(#${filterId})`}
            />
          </g>
        )}
      </svg>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <div style={{
          fontSize: '9px',
          color: 'rgba(255,255,255,0.32)',
          letterSpacing: '0.2px',
        }}>
          {label}{' '}
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
          <span style={{
            fontSize: '22px',
            fontWeight: 700,
            lineHeight: 1,
            letterSpacing: '-0.5px',
            color: '#e8ecea',
          }}>
            {displayVal}
          </span>
          {pct > 0 && sublabel && (
            <span style={{
              fontSize: '11px',
              color: 'rgba(255,255,255,0.25)',
              fontFamily: 'monospace',
            }}>
              /{sublabel}
            </span>
          )}
          {pct > 0 && !sublabel && (
            <span style={{
              fontSize: '11px',
              color: color,
              opacity: 0.6,
              fontFamily: 'monospace',
            }}>
              %
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
