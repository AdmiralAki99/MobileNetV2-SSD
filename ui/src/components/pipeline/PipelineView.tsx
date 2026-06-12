import { useState } from 'react'
import { ExperimentList } from './ExperimentList'
import { DetailPanel } from './DetailPanel'
import { PillButton } from '../PillButton'
import { usePolling } from '../../api/hooks'
import { fetchExperiments } from '../../api/client'
import { type Experiment } from '../../types/experiment'

interface Props {
  statusFilter: string
}

export const PipelineView = ({ statusFilter }: Props) => {
  const [selectedExp, setSelectedExp] = useState<Experiment | null>(null)

  const { data: experiments, loading, refresh } = usePolling<Experiment[]>(
    fetchExperiments as () => Promise<Experiment[]>,
    15000,
  )

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 210px', height: '100%', overflow: 'hidden' }}>
      <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '18px 18px 8px', flexShrink: 0 }}>
          <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.4px', margin: 0 }}>
            ML<br />Pipeline
          </h1>
          <p style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: 4 }}>
            {loading ? 'Loading…' : `${(experiments ?? []).length} experiments`}
          </p>
        </div>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <ExperimentList
            experiments={experiments ?? []}
            selectedId={selectedExp?.experiment_id}
            statusFilter={statusFilter}
            onSelect={setSelectedExp}
          />
        </div>
        <div style={{ padding: '8px 18px 14px', flexShrink: 0 }}>
          <PillButton onClick={refresh} style={{ width: '100%', justifyContent: 'center' }}>
            Refresh
          </PillButton>
        </div>
      </div>
      <DetailPanel selectedExp={selectedExp} onRefresh={refresh} />
    </div>
  )
}
