import { useState } from 'react'
import * as d3 from 'd3'
import { VOC_CLASSES } from './mockData'

const N = 21
const CELL = 16
const LABELS = ['bg', ...VOC_CLASSES.map(c => c.slice(0, 4))]

interface Props {
  matrix: number[][]
}

export const ConfusionMatrix = ({ matrix }: Props) => {
  const [tip, setTip] = useState<{ r: number; c: number; val: number; x: number; y: number } | null>(null)

  const flat = matrix.flat()
  const mx = Math.max(...flat, 1)
  const colSc = d3.scaleSequential(d3.interpolate('#0a1a14', '#00d4a0')).domain([0, mx])

  return (
    <div data-testid="confusion-matrix" style={{ padding: '20px 22px', display: 'flex', flexDirection: 'column', gap: 14, borderRadius: 14, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Confusion Matrix</span>
        <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>rows actual · cols predicted</span>
      </div>
      <div style={{ overflowX: 'auto', position: 'relative' }}>
        <svg
          data-testid="matrix-svg"
          width={N * CELL + 2}
          height={N * CELL + 2}
          style={{ display: 'block' }}
        >
          {matrix.map((row, r) =>
            row.map((val, c) => {
              const x = 1 + c * CELL, y = 1 + r * CELL
              return (
                <rect
                  key={`${r}-${c}`}
                  data-testid={`cell-${r}-${c}`}
                  x={x} y={y}
                  width={CELL - 1} height={CELL - 1}
                  rx="2"
                  fill={val > 0 ? colSc(val) : '#0a1410'}
                  onMouseEnter={e => {
                    const sr = e.currentTarget.closest('svg')!.getBoundingClientRect()
                    setTip({ r, c, val, x: e.clientX - sr.left, y: e.clientY - sr.top })
                  }}
                  onMouseLeave={() => setTip(null)}
                  style={{ cursor: 'default' }}
                />
              )
            })
          )}
          {tip && tip.val > 0 && (() => {
            const tx = Math.min(tip.x + 8, N * CELL - 118)
            const ty = Math.max(tip.y - 38, 2)
            return (
              <g data-testid="cell-tooltip">
                <rect x={tx} y={ty} width={114} height={36} rx="6"
                  fill="rgba(8,14,12,0.97)" stroke="rgba(255,255,255,0.14)" strokeWidth="0.8" />
                <text x={tx + 9} y={ty + 13} fontSize="8.5" fill="var(--text-secondary)" fontFamily="'DM Sans',sans-serif">
                  {LABELS[tip.r]} → {LABELS[tip.c]}
                </text>
                <text x={tx + 9} y={ty + 26} fontSize="10" fontWeight="700" fill="var(--text-primary)" fontFamily="'DM Sans',sans-serif">
                  {tip.val} samples
                </text>
              </g>
            )
          })()}
        </svg>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
        <span data-testid="legend-min" style={{ fontSize: '9px', color: 'var(--text-tertiary)' }}>0</span>
        <div style={{ flex: 1, height: 4, borderRadius: 3, background: 'linear-gradient(90deg,#0a1a14,#00d4a0)' }} />
        <span data-testid="legend-max" style={{ fontSize: '9px', color: 'var(--text-tertiary)' }}>{mx}</span>
      </div>
    </div>
  )
}
