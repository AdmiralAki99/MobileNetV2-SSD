interface Props {
  label: string
  value: string
  sub?: string
  accentColor?: string
  sparkData?: number[]
}

export const StatTile = ({ label, value, sub, accentColor, sparkData }: Props) => {
  const W = 72, H = 24, pad = 2
  const vals = sparkData ? sparkData.slice(-40) : []
  let spark = null

  if (vals.length > 1) {
    const mn = Math.min(...vals), mx = Math.max(...vals)
    const barW = Math.max(1, (W - pad * 2) / vals.length - 0.5)
    const xSc = (i: number) => pad + i * ((W - pad * 2) / vals.length)
    const ySc = (v: number) => mx === mn ? H / 2 : ((v - mn) / (mx - mn)) * (H - pad * 2)

    spark = (
      <svg
        data-testid="sparkline"
        width={W} height={H}
        style={{ position: 'absolute', bottom: 0, right: 0, opacity: 0.25 }}
        viewBox={`0 0 ${W} ${H}`}
      >
        {vals.map((v, i) => {
          const barH = Math.max(1, ySc(v))
          return (
            <rect key={i}
              x={xSc(i)} y={H - pad - barH}
              width={barW} height={barH}
              fill={accentColor ?? '#8a9a6a'}
            />
          )
        })}
      </svg>
    )
  }

  return (
    <div
      data-testid="stat-tile"
      style={{
        position: 'relative', overflow: 'hidden',
        padding: '18px 20px 16px',
      }}
    >
      {spark}

      <div style={{
        position: 'absolute', top: 0, left: 20, right: 20, height: '1px',
        background: `linear-gradient(90deg, transparent, ${accentColor ?? '#8a9a6a'}66, transparent)`,
      }} />

      <span
        data-testid="stat-label"
        style={{
          display: 'block',
          fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px',
          textTransform: 'uppercase', color: 'rgba(255,255,255,0.22)',
          marginBottom: 8,
        }}
      >
        {label}
      </span>

      <span
        data-testid="stat-value"
        style={{
          display: 'block',
          fontSize: '28px', fontWeight: 700,
          color: accentColor ?? '#dde2e0', letterSpacing: '-1.5px', lineHeight: 1,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </span>

      {sub && (
        <span
          data-testid="stat-sub"
          style={{
            display: 'block',
            fontSize: '9px', color: 'rgba(255,255,255,0.2)',
            marginTop: 6, fontFamily: 'monospace',
          }}
        >
          {sub}
        </span>
      )}
    </div>
  )
}
