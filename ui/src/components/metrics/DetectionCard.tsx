import { useState } from 'react'
import { BOX_COLORS } from './mockData'

const W = 200, H = 136

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
    <div data-testid={`detection-card-${img.id}`} style={{ flexShrink: 0, width: W, display: 'flex', flexDirection: 'column', gap: 6 }}>
      <svg width={W} height={H} style={{ display: 'block', borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.07)' }}>
        <rect width={W} height={H} fill="#0b1812" />
        {[1, 2, 3, 4].map(i => <line key={`v${i}`} x1={i * (W / 5)} y1={0} x2={i * (W / 5)} y2={H} stroke="rgba(255,255,255,0.025)" strokeWidth="1" />)}
        {[1, 2, 3].map(i => <line key={`h${i}`} x1={0} y1={i * (H / 4)} x2={W} y2={i * (H / 4)} stroke="rgba(255,255,255,0.025)" strokeWidth="1" />)}
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="11" fill="rgba(255,255,255,0.06)"
          fontFamily="'DM Sans',sans-serif" dy=".35em">{img.label}</text>
        {img.boxes.map((box, b) => {
          const bx = box.x * W, by = box.y * H, bw = box.w * W, bh = box.h * H
          const col = BOX_COLORS[b % BOX_COLORS.length]
          const isH = hov === b
          const lw = box.cls.length * 5.5 + 32
          return (
            <g key={b} data-testid={`box-${img.id}-${b}`}
              onMouseEnter={() => setHov(b)} onMouseLeave={() => setHov(null)}>
              <rect x={bx} y={by} width={bw} height={bh} rx="3"
                fill={`${col}${isH ? '25' : '10'}`} stroke={col} strokeWidth={isH ? 2 : 1.3}
                style={{ transition: 'all 0.15s' }} />
              <rect x={bx} y={Math.max(by - 14, 0)} width={lw} height={14} rx="3" fill={col} />
              <text x={bx + 4} y={Math.max(by - 3, 10)} fontSize="7.5" fill="#000" fontWeight="700"
                fontFamily="'DM Sans',sans-serif">{box.cls} {(box.score * 100).toFixed(0)}%</text>
            </g>
          )
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 2px' }}>
        <span data-testid={`card-label-${img.id}`} style={{ fontSize: '9.5px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{img.label}</span>
        <span data-testid={`card-count-${img.id}`} style={{ fontSize: '9.5px', color: 'var(--text-tertiary)' }}>{img.boxes.length} det</span>
      </div>
    </div>
  )
}
