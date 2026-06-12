interface Props {
  label: string
  value: string
  sub?: string
  accentColor?: string
  sparkData?: number[]
}

export const StatTile = ({ label, value, sub, accentColor, sparkData }: Props) => {
  const W = 80, H = 28, pad = 2
  const vals = sparkData ? sparkData.slice(-30) : []
  let spark = null

  if (vals.length > 1) {
    const mn = Math.min(...vals), mx = Math.max(...vals)
    const xSc = (i: number) => (i / (vals.length - 1)) * (W - pad * 2) + pad
    const ySc = (v: number) => H - pad - (mx === mn ? H / 2 : (v - mn) / (mx - mn) * (H - pad * 2))
    const pts = vals.map((v, i) => `${xSc(i)},${ySc(v)}`).join(' ')
    const area = `M${xSc(0)},${H} ` + vals.map((v, i) => `L${xSc(i)},${ySc(v)}`).join(' ') + ` L${xSc(vals.length - 1)},${H} Z`

    spark = (
      <svg
        data-testid="sparkline"
        width={W} height={H}
        style={{ position: 'absolute', bottom: 0, right: 0, opacity: 0.25 }}
        viewBox={`0 0 ${W} ${H}`}
      >
        <defs>
          <linearGradient id={`sg-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accentColor ?? 'var(--accent)'} stopOpacity="1" />
            <stop offset="100%" stopColor={accentColor ?? 'var(--accent)'} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#sg-${label})`} />
        <polyline points={pts} fill="none" stroke={accentColor ?? 'var(--accent)'} strokeWidth="1.2" />
      </svg>
    )
  }

  return (
    <div
      data-testid="stat-tile"
      style={{
        background: '#0f1714', border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 16, overflow: 'hidden', position: 'relative',
        padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 4,
      }}
    >
      {spark}
      <span data-testid="stat-label" style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.7px', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>
        {label}
      </span>
      <span data-testid="stat-value" style={{ fontSize: '28px', fontWeight: 700, color: accentColor ?? 'var(--text-primary)', letterSpacing: '-1px', lineHeight: 1.1 }}>
        {value}
      </span>
      {sub && (
        <span data-testid="stat-sub" style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginTop: 1 }}>
          {sub}
        </span>
      )}
    </div>
  )
}
