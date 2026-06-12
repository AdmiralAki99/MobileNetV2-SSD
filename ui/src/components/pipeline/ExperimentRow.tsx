import { StatusBadge } from '../StatusBadge'
import { type Experiment } from '../../types/experiment'

interface Props {
  exp: Experiment
  selected?: boolean
  onClick?: () => void
}

export const ExperimentRow = ({ exp, selected = false, onClick }: Props) => (
  <div
    role="button"
    aria-selected={selected}
    onClick={onClick}
    style={{
      padding: '10px 12px', borderRadius: 10, cursor: 'pointer', marginBottom: 6,
      border: `1px solid ${selected ? 'var(--accent)' : 'var(--border-subtle)'}`,
      background: selected ? 'rgba(0,212,160,0.06)' : 'var(--bg-surface)',
      transition: 'all 0.2s ease',
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
        {exp.experiment_id}
      </span>
      <StatusBadge status={exp.status} />
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
        {exp.fingerprint ?? '—'}
      </span>
      {exp.best_metric !== undefined && (
        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
          mAP {(exp.best_metric * 100).toFixed(1)}%
        </span>
      )}
      {exp.region && (
        <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{exp.region}</span>
      )}
    </div>
  </div>
)
