import { useState } from 'react'
import { ExperimentList } from './ExperimentList'
import { DetailPanel } from './DetailPanel'
import { ExperimentOrbit } from './ExperimentOrbit'
import { AWS_REGIONS } from './WorldGlobe'
import { DotMatrix } from './DotMatrix'
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

  const activeRegion = selectedExp?.region ?? null

  const regionCounts = (experiments ?? []).reduce<Record<string, number>>((acc, e) => {
    if (e.region) acc[e.region] = (acc[e.region] ?? 0) + 1
    return acc
  }, {})

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '200px 1fr 210px',
        height: '100%',
        overflow: 'hidden',
        background: `
          radial-gradient(ellipse 70% 60% at 50% 50%, rgba(0,180,120,0.05) 0%, transparent 70%),
          var(--bg-primary)
        `,
      }}
    >

      <div style={{
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        borderRight: '1px solid rgba(255,255,255,0.04)',
      }}>
        <div style={{ padding: '18px 16px 10px', flexShrink: 0 }}>
          <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1.4px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 6 }}>
            ML Pipeline
          </div>
          <div style={{ fontSize: '22px', fontWeight: 700, color: '#dde2e0', letterSpacing: '-0.8px', lineHeight: 1.1 }}>
            Experiments
          </div>
          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', marginTop: 5, fontFamily: 'monospace' }}>
            {loading ? 'Loading…' : `${(experiments ?? []).length} experiments`}
          </div>
        </div>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <ExperimentList
            experiments={experiments ?? []}
            selectedId={selectedExp?.experiment_id}
            statusFilter={statusFilter}
            onSelect={setSelectedExp}
          />
        </div>
        <div style={{ padding: '8px 16px 14px', flexShrink: 0 }}>
          <PillButton onClick={refresh} style={{ width: '100%', justifyContent: 'center' }}>
            Refresh
          </PillButton>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        <div style={{ flex: 1, overflow: 'hidden', padding: '12px' }}>
          <ExperimentOrbit
            experiments={experiments ?? []}
            selectedId={selectedExp?.experiment_id}
            onSelect={setSelectedExp}
          />
        </div>

        <div style={{
          flexShrink: 0,
          borderTop: '1px solid rgba(255,255,255,0.04)',
        }}>
          <DotMatrix
            experiments={experiments ?? []}
            selectedId={selectedExp?.experiment_id}
            onSelect={setSelectedExp}
          />
        </div>

        <div style={{
          flexShrink: 0, padding: '10px 20px 14px',
          borderTop: '1px solid rgba(255,255,255,0.04)',
          display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center',
        }}>
          <span style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', flexShrink: 0 }}>
            AWS Regions
          </span>
          {AWS_REGIONS.filter(r => regionCounts[r.id]).map(r => (
            <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{
                width: 5, height: 5,
                background: r.id === activeRegion ? '#00d4a0' : 'rgba(0,200,140,0.5)',
                boxShadow: r.id === activeRegion ? '0 0 6px rgba(0,212,160,0.8)' : 'none',
              }} />
              <span style={{ fontSize: '9px', fontFamily: 'monospace', color: r.id === activeRegion ? 'rgba(0,212,160,0.9)' : 'rgba(255,255,255,0.25)' }}>
                {r.id}
              </span>
              {regionCounts[r.id] && (
                <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.15)', fontFamily: 'monospace' }}>
                  ×{regionCounts[r.id]}
                </span>
              )}
            </div>
          ))}
          {Object.keys(regionCounts).length === 0 && AWS_REGIONS.slice(0, 6).map(r => (
            <div key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 4, height: 4, background: 'rgba(0,200,140,0.3)' }} />
              <span style={{ fontSize: '9px', fontFamily: 'monospace', color: 'rgba(255,255,255,0.18)' }}>
                {r.id}
              </span>
            </div>
          ))}
          <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.12)', fontFamily: 'monospace', marginLeft: 'auto' }}>
            click a node to inspect
          </span>
        </div>
      </div>

      <div style={{ borderLeft: '1px solid rgba(255,255,255,0.04)', overflow: 'hidden' }}>
        <DetailPanel selectedExp={selectedExp} onRefresh={refresh} />
      </div>

    </div>
  )
}
