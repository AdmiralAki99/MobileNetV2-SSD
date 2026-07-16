import { useState } from 'react'
import * as d3 from 'd3'
import { VOC_CLASSES } from './mockData'

const N = 21
const CELL = 16
const LABELS = ['bg', ...VOC_CLASSES.map(c => c.slice(0, 4))]

const COL_HI = '#8a9a6a'
const COL_LO = '#4a5a6a'
const COL_EMPTY = 'rgba(255,255,255,0.03)'

interface Props {
  matrix: number[][]
}

export const ConfusionMatrix = ({ matrix }: Props) => {
  const [tip, setTip] = useState<{ r: number; c: number; val: number; x: number; y: number } | null>(null)

  const flat = matrix.flat()
  const mx = Math.max(...flat, 1)

  const diagSc = d3.scaleSequential(d3.interpolate('rgba(138,154,106,0.08)', COL_HI)).domain([0, mx])
  const errSc  = d3.scaleSequential(d3.interpolate('rgba(74,90,106,0.06)',   COL_LO)).domain([0, mx])

  return (
    <div
      data-testid="confusion-matrix"
      style={{
        padding: '18px 20px',
        display: 'flex', flexDirection: 'column', gap: 12,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.6px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)' }}>
          Confusion Matrix
        </span>
        <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.2)' }}>rows actual · cols predicted</span>
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
              const isDiag = r === c
              const fill = val > 0
                ? (isDiag ? diagSc(val) : errSc(val))
                : COL_EMPTY
              return (
                <rect
                  key={`${r}-${c}`}
                  data-testid={`cell-${r}-${c}`}
                  x={x} y={y}
                  width={CELL - 1} height={CELL - 1}
                  rx="1"
                  fill={fill}
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
            const ty = Math.max(tip.y - 46, 2)
            return (
              <g data-testid="cell-tooltip">
                <rect x={tx} y={ty} width={114} height={40} rx="4"
                  fill="rgba(6,10,9,0.97)" stroke="rgba(255,255,255,0.1)" strokeWidth="0.8" />
                <text x={tx + 9} y={ty + 14} fontSize="8.5" fill="rgba(255,255,255,0.35)" fontFamily="monospace">
                  {LABELS[tip.r]} → {LABELS[tip.c]}
                </text>
                <text x={tx + 9} y={ty + 30} fontSize="10" fontWeight="700" fill="#e8eae9" fontFamily="monospace">
                  {tip.val} samples
                </text>
              </g>
            )
          })()}
        </svg>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 2 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 40, height: 3, background: `linear-gradient(90deg, rgba(138,154,106,0.08), ${COL_HI})` }} />
          <span data-testid="legend-min" style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>0</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 40, height: 3, background: `linear-gradient(90deg, rgba(74,90,106,0.06), ${COL_LO})` }} />
          <span data-testid="legend-max" style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{mx}</span>
        </div>
        <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.15)', fontFamily: 'monospace', marginLeft: 'auto' }}>
          max {mx}
        </span>
      </div>
    </div>
  )
}
