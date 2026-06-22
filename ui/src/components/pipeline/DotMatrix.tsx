import { useState } from 'react'
import { type Experiment } from '../../types/experiment'

const STATUS_COLOR: Record<string, string> = {
  running:  '#00d4a0',
  complete: '#8a9a6a',
  failed:   '#c87070',
  pending:  'rgba(255,255,255,0.35)',
  stopped:  'rgba(255,255,255,0.2)',
}

const STATUS_ORDER = ['running', 'pending', 'complete', 'failed', 'stopped']

const DOT = 5
const GAP = 3

interface Props {
  experiments: Experiment[]
  selectedId?: string
  onSelect: (exp: Experiment) => void
}

export const DotMatrix = ({ experiments, selectedId, onSelect }: Props) => {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const groups = STATUS_ORDER.reduce<Record<string, Experiment[]>>((acc, s) => {
    acc[s] = experiments.filter(e => e.status === s)
    return acc
  }, {})

  const known = new Set(STATUS_ORDER)
  const others = experiments.filter(e => !known.has(e.status))
  if (others.length) groups['other'] = others

  const rows = Object.entries(groups).filter(([, exps]) => exps.length > 0)

  const selectedStatus = experiments.find(e => e.experiment_id === selectedId)?.status ?? null

  const tip = hoveredId ? experiments.find(e => e.experiment_id === hoveredId) : null

  const LABEL_W = 52
  const COLS    = 18

  return (
    <div
      data-testid="dot-matrix"
      style={{ padding: '10px 16px 14px', position: 'relative' }}
    >
      <div style={{
        fontSize: '8px',
        fontWeight: 600,
        letterSpacing: '1px',
        textTransform: 'uppercase',
        color: 'rgba(255,255,255,0.14)',
        marginBottom: 10,
      }}>
        360 View by Blocks
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {rows.map(([status, exps]) => {
          const color      = STATUS_COLOR[status] ?? 'rgba(255,255,255,0.2)'
          const isSelRow   = status === selectedStatus

          return (
            <div key={status} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>

              <div style={{
                width: LABEL_W,
                flexShrink: 0,
                fontSize: '8.5px',
                textTransform: 'capitalize',
                fontFamily: 'monospace',
                color: isSelRow ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.16)',
                transition: 'color 0.2s',
              }}>
                {status}
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${COLS}, ${DOT}px)`,
                gap: `${GAP}px`,
                padding: '4px',
                borderRadius: 3,
                border: isSelRow
                  ? '1px solid rgba(255,255,255,0.2)'
                  : '1px solid transparent',
                background: isSelRow
                  ? 'rgba(255,255,255,0.018)'
                  : 'transparent',
                transition: 'border-color 0.2s, background 0.2s',
              }}>
                {exps.map(exp => {
                  const isSel     = exp.experiment_id === selectedId
                  const isHov     = exp.experiment_id === hoveredId
                  const isRunning = exp.status === 'running'

                  return (
                    <div
                      key={exp.experiment_id}
                      data-testid={`dot-${exp.experiment_id}`}
                      onClick={() => onSelect(exp)}
                      onMouseEnter={() => setHoveredId(exp.experiment_id)}
                      onMouseLeave={() => setHoveredId(null)}
                      style={{
                        width: DOT,
                        height: DOT,
                        borderRadius: 1,
                        background: color,
                        opacity: isSel ? 1 : isHov ? 0.65 : isRunning ? 0.5 : 0.1,
                        boxShadow: isSel
                          ? `0 0 0 1px rgba(255,255,255,0.45), 0 0 5px ${color}99`
                          : 'none',
                        cursor: 'pointer',
                        transition: 'opacity 0.12s, box-shadow 0.12s',
                        animation: isRunning ? 'dm-pulse 2.4s ease-in-out infinite' : 'none',
                        flexShrink: 0,
                      }}
                    />
                  )
                })}
              </div>
            </div>
          )
        })}

        {experiments.length === 0 && (
          <div style={{
            paddingLeft: LABEL_W + 8,
            fontSize: '9px',
            color: 'rgba(255,255,255,0.1)',
            fontFamily: 'monospace',
          }}>
            no experiments
          </div>
        )}
      </div>

      {tip && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: LABEL_W + 24,
          marginBottom: 8,
          backdropFilter: 'blur(16px) saturate(1.4)',
          background: 'rgba(4,10,7,0.94)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 5,
          padding: '6px 10px',
          pointerEvents: 'none',
          zIndex: 40,
          minWidth: 150,
        }}>
          <div style={{
            fontSize: '9.5px',
            fontWeight: 700,
            color: '#dde2e0',
            fontFamily: 'monospace',
            marginBottom: 4,
          }}>
            {tip.experiment_id}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              fontSize: '8.5px',
              color: STATUS_COLOR[tip.status] ?? '#fff',
              fontFamily: 'monospace',
            }}>
              {tip.status}
            </span>
            {tip.best_metric != null && (
              <span style={{
                fontSize: '8.5px',
                color: 'rgba(255,255,255,0.3)',
                fontFamily: 'monospace',
              }}>
                mAP {(tip.best_metric * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      )}

      <style>{`
        @keyframes dm-pulse {
          0%, 100% { opacity: 0.38; }
          50%       { opacity: 0.75; }
        }
      `}</style>
    </div>
  )
}
