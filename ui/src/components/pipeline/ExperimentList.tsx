import { ExperimentRow } from './ExperimentRow'
import { type Experiment } from '../../types/experiment'

interface Props {
  experiments: Experiment[]
  selectedId?: string
  statusFilter: string
  onSelect: (exp: Experiment) => void
}

export const ExperimentList = ({ experiments, selectedId, statusFilter, onSelect }: Props) => {
  const filtered = statusFilter === 'all'
    ? experiments
    : experiments.filter(e => e.status === statusFilter)

  return (
    <div style={{ padding: '12px 8px', overflowY: 'auto', height: '100%' }}>
      {filtered.length === 0 && (
        <div style={{ padding: '24px 12px', textAlign: 'center', fontSize: '12px', color: 'var(--text-tertiary)' }}>
          No experiments match this filter.
        </div>
      )}
      {filtered.map(exp => (
        <ExperimentRow
          key={exp.experiment_id}
          exp={exp}
          selected={exp.experiment_id === selectedId}
          onClick={() => onSelect(exp)}
        />
      ))}
    </div>
  )
}
