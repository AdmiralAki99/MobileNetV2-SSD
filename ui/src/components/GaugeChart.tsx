import { useState, useEffect } from 'react'

interface Props {
  value: number
  max: number
  size?: number
  greenPct?: number
  label?: string
  color?: string
}

export const GaugeChart = ({ value, max, size = 90, greenPct = 0.7 }: Props) => {
  const [animVal, setAnimVal] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setAnimVal(value), 400)
    return () => clearTimeout(t)
  }, [value])

  const cx = 55, cy = 55, r = 42
  const startA = 135, totalA = 270
  const pct = Math.min(animVal / max, 1)
  const toRad = (d: number) => (d - 90) * Math.PI / 180
  const ptOn = (a: number) => ({ x: cx + r * Math.cos(toRad(a)), y: cy + r * Math.sin(toRad(a)) })

  const makeArc = (from: number, to: number) => {
    const s = ptOn(to), e = ptOn(from)
    const large = to - from > 180 ? 1 : 0
    return `M${s.x} ${s.y} A${r} ${r} 0 ${large} 0 ${e.x} ${e.y}`
  }

  const needleA = startA + pct * totalA
  const nTip = ptOn(needleA)
  const greenEnd = startA + greenPct * totalA

  return (
    <div style={{ textAlign: 'center' }}>
      <svg
        role="img"
        aria-label={`gauge: ${value} of ${max}`}
        width={size} height={size * 0.75} viewBox="0 0 110 82"
      >
        <path d={makeArc(startA, startA + totalA)} fill="none" stroke="var(--bg-pill)" strokeWidth="5" strokeLinecap="round"/>
        <path d={makeArc(startA, greenEnd)} fill="none" stroke="#65c16a" strokeWidth="5" strokeLinecap="round"/>
        <path d={makeArc(greenEnd, startA + totalA)} fill="none" stroke="#e84855" strokeWidth="5" strokeLinecap="round"/>
        <line x1={cx} y1={cy} x2={nTip.x} y2={nTip.y}
          stroke="var(--text-primary)" strokeWidth="1.5" strokeLinecap="round"
          style={{ transition: 'all 1.2s cubic-bezier(0.34,1.56,0.64,1)' }}/>
        <circle cx={cx} cy={cy} r="3" fill="var(--bg-surface)" stroke="var(--text-tertiary)" strokeWidth="1"/>
      </svg>
    </div>
  )
}
