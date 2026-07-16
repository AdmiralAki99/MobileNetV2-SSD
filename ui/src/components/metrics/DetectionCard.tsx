import { useState } from 'react'

const W = 200, H = 136

const MUTED_COLS = ['#8a9a6a', '#6a7a8a', '#9a8a6a', '#6a8a7a', '#7a6a8a', '#8a7a6a']

export interface DetectionBox {
  x: number; y: number; w: number; h: number
  cls: string
  score: number
}

export interface DetectionImage {
  id: number | string
  label: string
  boxes: DetectionBox[]
}

interface Props { img: DetectionImage }

export const DetectionCard = ({ img }: Props) => {
  const [hov, setHov] = useState<number | null>(null)

  return (
    <div
      data-testid={`detection-card-${img.id}`}
      style={{ flexShrink: 0, width: W, display: 'flex', flexDirection: 'column', gap: 5 }}
    >
      <svg
        width={W} height={H}
        style={{ display: 'block', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8 }}
      >
        <rect width={W} height={H} fill="rgba(10,18,14,0.55)" />

        {[1, 2, 3, 4].map(i => (
          <line key={`v${i}`} x1={i * (W / 5)} y1={0} x2={i * (W / 5)} y2={H}
            stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
        ))}
        {[1, 2, 3].map(i => (
          <line key={`h${i}`} x1={0} y1={i * (H / 4)} x2={W} y2={i * (H / 4)}
            stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
        ))}

        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.04)"
          fontFamily="monospace" dy=".35em">{img.label}</text>

        {img.boxes.map((box, b) => {
          const bx = box.x * W, by = box.y * H, bw = box.w * W, bh = box.h * H
          const col = MUTED_COLS[b % MUTED_COLS.length]
          const isH = hov === b
          const labelText = `${box.cls} ${(box.score * 100).toFixed(0)}%`
          const lw = labelText.length * 5 + 8
          const ly = by > 14 ? by - 14 : by + bh
          return (
            <g key={b} data-testid={`box-${img.id}-${b}`}
              onMouseEnter={() => setHov(b)} onMouseLeave={() => setHov(null)}>
              <rect x={bx} y={by} width={bw} height={bh}
                fill={isH ? `${col}22` : 'transparent'}
                stroke={col}
                strokeWidth={isH ? 2 : 1.3}
                opacity={isH ? 1 : 0.65}
                style={{ transition: 'all 0.12s' }}
              />
              <rect x={bx} y={ly} width={lw} height={13} fill={col} opacity={isH ? 0.9 : 0.6} />
              <text x={bx + 4} y={ly + 9} fontSize="7" fill="rgba(0,0,0,0.85)" fontWeight="700"
                fontFamily="monospace">{labelText}</text>
              <circle cx={bx} cy={by} r="2.5" fill="#fff" opacity={isH ? 0.7 : 0.3} />
            </g>
          )
        })}
      </svg>

      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 2px' }}>
        <span
          data-testid={`card-label-${img.id}`}
          style={{ fontSize: '9px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}
        >
          {img.label}
        </span>
        <span
          data-testid={`card-count-${img.id}`}
          style={{ fontSize: '9px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}
        >
          {img.boxes.length} det
        </span>
      </div>
    </div>
  )
}
